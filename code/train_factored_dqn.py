"""Factored Double DQN for Role A.

Unlike the original flat grid MLP, this Q-network shares assignment and charging
heads across entities and receives routed-distance, deadline, and battery
features. It remains a value-based, replay-buffer, target-network Double DQN.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.evaluate import evaluate
from role_b.features import RoutedCache, extract_features
from role_b.networks import FactoredActorCritic
from role_b.utils import batch_features, single_batch
from role_c.role_c_rollout import RoleCRolloutPlanner


class FactoredDQNPolicy:
    def __init__(self, net, cfg, device="cpu"):
        self.net = net.to(device).eval()
        self.cfg = cfg
        self.device = torch.device(device)
        self.cache = RoutedCache(cfg.neighborhood)

    def act(self, obs):
        feat = extract_features(obs, self.cfg, self.cache)
        with torch.no_grad():
            q, _ = self.net(single_batch(feat, self.device))
        return int(q[0].argmax().item())

    def action_values(self, obs):
        feat = extract_features(obs, self.cfg, self.cache)
        with torch.no_grad():
            q, _ = self.net(single_batch(feat, self.device))
        values = q[0].cpu().numpy()
        return np.where(np.asarray(obs["action_mask"], bool), values, np.nan)

    def action_probs(self, obs):
        return None

    def state_values(self, obs):
        return None


def save_policy(path, net, cfg, params):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "hidden": int(params["hidden"]),
        "env_config": asdict(cfg),
        "train_config": params,
    }, path)


def load_policy(path, device="cpu", cfg_override=None):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg_override or Config.from_dict(ck["env_config"])
    net = FactoredActorCritic(hidden=int(ck["hidden"]))
    net.load_state_dict(ck["model_state"])
    return FactoredDQNPolicy(net, cfg, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/factored_double_dqn.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    params = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(params["seed"] if args.seed is None else args.seed)
    params = dict(params)
    params["seed"] = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)

    cfg = Config.from_yaml(params["env_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = int(params["hidden"])
    net = FactoredActorCritic(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(params["learning_rate"]))
    replay = deque(maxlen=int(params["buffer_size"]))
    rng = np.random.default_rng(seed)
    env = DroneDispatchEnv(cfg)
    cache = RoutedCache(cfg.neighborhood)

    demo_steps = int(params.get("teacher_pretrain_steps", 0))
    if demo_steps > 0:
        demo_env = DroneDispatchEnv(cfg)
        teacher = RoleCRolloutPlanner.from_yaml(
            cfg, params.get("teacher_config", "configs/role_c_rollout.yaml"),
            depth=int(params.get("teacher_depth", 1)))
        demo_cache = RoutedCache(cfg.neighborhood)
        demos = []
        for episode in range(int(params.get("teacher_episodes", 30))):
            demo_obs, _ = demo_env.reset(seed=900_000 + seed * 10_000 + episode)
            done = False
            while not done:
                action = teacher.act(demo_obs)
                demos.append((extract_features(demo_obs, cfg, demo_cache), action))
                demo_obs, _, term, trunc, _ = demo_env.step(action)
                done = term or trunc
        demo_opt = torch.optim.Adam(
            net.parameters(), lr=float(params.get("teacher_learning_rate", 3e-4)))
        demo_batch = int(params.get("teacher_batch_size", 128))
        for update in range(1, demo_steps + 1):
            sample = random.sample(demos, min(demo_batch, len(demos)))
            b = batch_features([x[0] for x in sample], device)
            actions = torch.as_tensor(
                [x[1] for x in sample], dtype=torch.int64, device=device)
            q, _ = net(b)
            loss = F.cross_entropy(q, actions)
            demo_opt.zero_grad(); loss.backward(); demo_opt.step()
            if update % max(1, demo_steps // 5) == 0:
                print(f"teacher pretrain {update}/{demo_steps} "
                      f"loss={loss.item():.4f}", flush=True)
        print(f"teacher demonstrations={len(demos)}", flush=True)

    target = copy.deepcopy(net)

    total_steps = int(params["total_steps"])
    warmup = int(params["learning_starts"])
    batch_size = int(params["batch_size"])
    gamma = float(params["gamma"])
    reward_scale = float(params["reward_scale"])
    eps_decay = int(params["epsilon_decay_steps"])
    eval_every = int(params["eval_every"])
    eval_seeds = [int(s) for s in params["eval_seeds"]]
    target_every = int(params["target_update_every"])
    train_every = int(params["train_every"])

    log_path = Path(params["log_path"].format(seed=seed))
    weight_path = Path(params["weight_path"].format(seed=seed))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    best_cost = float("inf")
    rows = []

    if demo_steps > 0:
        result = evaluate(
            FactoredDQNPolicy(net, cfg, device), cfg, eval_seeds)["mean"]
        rows.append({
            "seed": seed,
            "step": 0,
            "epsilon": 0.0,
            "cost_per_order": result["cost_per_order"],
            "success_rate": result["success_rate"],
            "episode_return": result["episode_return"],
        })
        best_cost = result["cost_per_order"]
        save_policy(weight_path, net, cfg, params)
        print({"teacher_warm_start": rows[-1]}, flush=True)

    obs, _ = env.reset(seed=seed * 100_000)
    episode = 0
    for step in range(1, total_steps + 1):
        frac = min(step / max(eps_decay, 1), 1.0)
        eps = float(params["epsilon_start"]) + frac * (
            float(params["epsilon_end"]) - float(params["epsilon_start"]))
        feat = extract_features(obs, cfg, cache)
        valid = np.flatnonzero(np.asarray(obs["action_mask"], dtype=bool))
        if rng.random() < eps:
            action = int(rng.choice(valid))
        else:
            with torch.no_grad():
                q, _ = net(single_batch(feat, device))
            action = int(q[0].argmax().item())

        nobs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        nfeat = extract_features(nobs, cfg, cache)
        replay.append((feat, action, reward / reward_scale, nfeat, float(done)))
        obs = nobs
        if done:
            episode += 1
            obs, _ = env.reset(seed=seed * 100_000 + episode)

        if len(replay) >= max(warmup, batch_size) and step % train_every == 0:
            sample = random.sample(replay, batch_size)
            b = batch_features([x[0] for x in sample], device)
            actions = torch.as_tensor(
                [x[1] for x in sample], dtype=torch.int64, device=device)
            rewards = torch.as_tensor(
                [x[2] for x in sample], dtype=torch.float32, device=device)
            nb = batch_features([x[3] for x in sample], device)
            dones = torch.as_tensor(
                [x[4] for x in sample], dtype=torch.float32, device=device)

            q, _ = net(b)
            chosen = q.gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                online_next, _ = net(nb)
                next_actions = online_next.argmax(1, keepdim=True)
                target_next, _ = target(nb)
                next_q = target_next.gather(1, next_actions).squeeze(1)
                td = rewards + gamma * (1.0 - dones) * next_q
            loss = F.smooth_l1_loss(chosen, td)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()

        if step % target_every == 0:
            target.load_state_dict(net.state_dict())

        if step % eval_every == 0 or step == total_steps:
            result = evaluate(
                FactoredDQNPolicy(net, cfg, device), cfg, eval_seeds)["mean"]
            row = {
                "seed": seed,
                "step": step,
                "epsilon": eps,
                "cost_per_order": result["cost_per_order"],
                "success_rate": result["success_rate"],
                "episode_return": result["episode_return"],
            }
            rows.append(row)
            print(row, flush=True)
            if result["cost_per_order"] < best_cost:
                best_cost = result["cost_per_order"]
                save_policy(weight_path, net, cfg, params)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"best cost={best_cost:.4f} -> {weight_path}")


if __name__ == "__main__":
    main()
