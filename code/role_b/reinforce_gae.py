"""REINFORCE with a learned value baseline and GAE(lambda) advantages.

On-policy Monte-Carlo policy gradient (Williams, 1992) where the high-variance
return is replaced by a GAE(lambda) advantage (Schulman et al., 2016) computed
against a learned critic baseline. One gradient step per batch of complete
episodes. This is the stepping stone to A2C (which bootstraps instead of waiting
for full episodes).

Checkpoints are selected on validation `cost_per_order` (the graded metric), NOT
on training return — the two differ because return includes the +10/+5 delivery
bonuses that the cost metric excludes.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.baselines import make_baseline

from .features import RoutedCache, extract_features
from .networks import FactoredActorCritic
from .rollout import compute_gae, normalize
from .utils import (CSVLogger, batch_features, evaluate_dispatch_agent,
                    seed_everything, single_batch)
from .adapters import PolicyGradientAgent


def _collect(env, net, cfg, device, cache, seed_rng, min_steps, reward_scale):
    """Collect complete episodes until >= min_steps transitions are gathered.
    Rewards stored for learning are scaled (keeps value targets on a sane scale);
    episode returns are tracked unscaled for logging."""
    feats, actions, rewards, dones = [], [], [], []
    ep_returns = []
    while len(actions) < min_steps:
        train_seed = int(seed_rng.integers(10_000, 5_000_000))
        obs, _ = env.reset(seed=train_seed)
        done = False
        ep_ret = 0.0
        while not done:
            feat = extract_features(obs, cfg, cache)
            b = single_batch(feat, device)
            with torch.no_grad():
                logits, _ = net(b)
            dist = torch.distributions.Categorical(logits=logits[0])
            a = int(dist.sample().item())
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            feats.append(feat)
            actions.append(a)
            rewards.append(r * reward_scale)
            dones.append(1.0 if done else 0.0)
            ep_ret += r
            obs = nobs
        ep_returns.append(ep_ret)
    return feats, actions, rewards, dones, ep_returns


def _bc_warmup(env, net, cfg, device, cache, seed_rng, epochs, dataset_steps, lr, minibatch=2048):
    """Behavior-clone greedy_nearest into the policy before RL.

    A random policy cold-starts and can (seed-dependently) fail to ever discover
    deliveries; cloning greedy first puts every seed in the delivering regime, so
    REINFORCE then reliably refines the policy below greedy instead of getting
    stuck in a no-delivery / half-delivery local optimum.
    """
    greedy = make_baseline("greedy_nearest", cfg)
    feats, acts = [], []
    while len(acts) < dataset_steps:
        obs, _ = env.reset(seed=int(seed_rng.integers(10_000, 5_000_000)))
        done = False
        while not done:
            ga = int(greedy.act(obs))
            feats.append(extract_features(obs, cfg, cache))
            acts.append(ga)
            obs, _, term, trunc, _ = env.step(ga)
            done = term or trunc
    acts_t = torch.as_tensor(acts, dtype=torch.long, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(acts)
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, minibatch):
            mb = perm[s:s + minibatch]
            b = batch_features([feats[i] for i in mb], device)
            logits, _ = net(b)
            loss = F.cross_entropy(logits, acts_t[mb])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


def train(cfg, params: dict, seed: int, log_path: str, weight_path: str) -> dict:
    p = params
    device = torch.device("cpu")
    torch.set_num_threads(1)   # 1 core per process so seeds run in parallel
    seed_everything(seed)

    lr = float(p.get("lr", 3e-4))
    gamma = float(p.get("gamma", 0.99))
    lam = float(p.get("gae_lambda", 0.95))
    ent_coef = float(p.get("ent_coef", 0.01))
    vf_coef = float(p.get("vf_coef", 0.5))
    hidden = int(p.get("hidden", 128))
    max_grad_norm = float(p.get("max_grad_norm", 0.5))
    norm_adv = bool(p.get("norm_adv", True))
    reward_scale = float(p.get("reward_scale", 0.1))
    steps_per_update = int(p.get("steps_per_update", 2048))
    total_steps = int(p.get("total_steps", 500_000))
    eval_every = int(p.get("eval_every_updates", 10))
    val_seeds = list(p.get("val_seeds", [200, 201, 202, 203, 204]))
    value_epochs = int(p.get("value_epochs", 1))   # extra critic fits/update (better baseline)
    lr_final_frac = float(p.get("lr_final_frac", 1.0))         # linear lr decay to lr*frac (1.0=off)
    ent_coef_final = float(p.get("ent_coef_final", ent_coef))  # entropy-coef schedule endpoint
    bc_warmup_updates = int(p.get("bc_warmup_updates", 0))     # clone greedy before RL (0=off)
    bc_lr = float(p.get("bc_lr", 1e-3))                        # supervised lr for the clone phase
    bc_dataset_steps = int(p.get("bc_dataset_steps", 40000))   # greedy transitions collected once

    os.makedirs(os.path.dirname(weight_path) or ".", exist_ok=True)
    env = DroneDispatchEnv(cfg)
    net = FactoredActorCritic(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    cache = RoutedCache(cfg.neighborhood)
    seed_rng = np.random.default_rng(seed)
    logger = CSVLogger(log_path)

    n_updates = max(1, total_steps // steps_per_update)
    global_steps = 0
    best_cost = float("inf")

    if bc_warmup_updates > 0:
        _bc_warmup(env, net, cfg, device, cache, seed_rng, bc_warmup_updates, bc_dataset_steps, bc_lr)

    for update in range(1, n_updates + 1):
        frac = (update - 1) / max(1, n_updates - 1)              # 0 -> 1 across training
        cur_lr = lr * (1.0 + frac * (lr_final_frac - 1.0))       # linear lr decay (anti-drift)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        cur_ent = ent_coef + frac * (ent_coef_final - ent_coef)  # entropy schedule (explore->exploit)
        feats, actions, rewards, dones, ep_returns = _collect(
            env, net, cfg, device, cache, seed_rng, steps_per_update, reward_scale)
        global_steps += len(actions)

        b = batch_features(feats, device)
        acts = torch.as_tensor(actions, dtype=torch.long, device=device)
        logits, values = net(b)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(acts)
        entropy = dist.entropy().mean()

        values_np = values.detach().cpu().numpy()
        adv, returns = compute_gae(rewards, values_np, dones, gamma, lam, 0.0)
        if norm_adv:
            adv = normalize(adv)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        policy_loss = -(logp * adv_t).mean()
        value_loss = F.smooth_l1_loss(values, ret_t)
        loss = policy_loss + vf_coef * value_loss - cur_ent * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()

        # Extra critic-only fits: a better value baseline lowers advantage variance
        # (the actor heads are separate params, so the policy is unchanged here).
        for _ in range(value_epochs - 1):
            _, v_extra = net(b)
            v_loss_extra = F.smooth_l1_loss(v_extra, ret_t)
            opt.zero_grad()
            (vf_coef * v_loss_extra).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()

        if update % eval_every == 0 or update == n_updates:
            agent = PolicyGradientAgent(net, cfg, device, deterministic=True)
            m = evaluate_dispatch_agent(agent, cfg, val_seeds)
            net.train()
            row = {
                "update": update, "step": global_steps,
                "train_return": float(np.mean(ep_returns)),
                "cost_per_order": m["cost_per_order"],
                "success_rate": m["success_rate"],
                "ontime_rate": m["ontime_rate"],
                "depletion_events": m["depletion_events"],
                "n_delivered": m["n_delivered"],
                "eval_return": m["episode_return"],
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
            }
            logger.log(row)
            if m["cost_per_order"] < best_cost:
                best_cost = m["cost_per_order"]
                torch.save({"state_dict": net.state_dict(), "hidden": hidden,
                            "cost_per_order": best_cost}, weight_path)
            print(f"[reinforce seed={seed}] upd {update}/{n_updates} "
                  f"step {global_steps} cost/order {m['cost_per_order']:.3f} "
                  f"(best {best_cost:.3f}) ret {np.mean(ep_returns):.1f} "
                  f"deliv {m['n_delivered']:.0f}", flush=True)

    logger.close()
    return {"best_cost": best_cost}
