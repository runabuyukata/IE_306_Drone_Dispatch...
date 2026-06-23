from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from dqn_agent import build_q_net, obs_to_vector, save_checkpoint
from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, next_mask, done):
        self.data.append((obs, action, reward, next_obs, next_mask, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        obs, actions, rewards, next_obs, next_masks, dones = zip(*batch)
        return (
            np.stack(obs),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.stack(next_masks),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.data)


def epsilon_by_step(step: int, cfg: dict) -> float:
    start = float(cfg["epsilon_start"])
    end = float(cfg["epsilon_end"])
    decay = max(int(cfg["epsilon_decay_steps"]), 1)
    frac = min(step / decay, 1.0)
    return start + frac * (end - start)


def choose_action(q_net, obs, cfg: Config, eps: float, device: torch.device) -> int:
    mask = np.asarray(obs["action_mask"], dtype=bool)
    valid = np.flatnonzero(mask)
    if random.random() < eps:
        return int(np.random.choice(valid))
    x = torch.as_tensor(obs_to_vector(obs, cfg), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q = q_net(x).squeeze(0).cpu().numpy()
    return int(np.argmax(np.where(mask, q, -np.inf)))


def train(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = Config.from_yaml(train_cfg["env_config"])
    env = DroneDispatchEnv(env_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = build_q_net(env_cfg, train_cfg["hidden_sizes"], dueling=bool(train_cfg["dueling"])).to(device)
    target_net = build_q_net(env_cfg, train_cfg["hidden_sizes"], dueling=bool(train_cfg["dueling"])).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=float(train_cfg["learning_rate"]))
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(int(train_cfg["buffer_size"]))

    Path(train_cfg["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(train_cfg["weight_path"]).parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    rows = []
    for episode in range(int(train_cfg["episodes"])):
        obs, _ = env.reset(seed=seed * 100000 + episode)
        ep_return = 0.0
        last_loss = ""
        done = False
        action_counts = {"assign": 0, "charge": 0, "noop": 0}
        charge_open_steps = 0
        for _ in range(int(train_cfg["max_steps_per_episode"])):
            eps = epsilon_by_step(global_step, train_cfg)
            charge_indices = [env_cfg.charge_index(d) for d in range(env_cfg.n_drones)]
            if np.asarray(obs["action_mask"], dtype=bool)[charge_indices].any():
                charge_open_steps += 1
            action = choose_action(q_net, obs, env_cfg, eps, device)
            action_counts[env_cfg.decode(action)[0]] += 1
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.add(
                obs_to_vector(obs, env_cfg),
                action,
                reward,
                obs_to_vector(next_obs, env_cfg),
                np.asarray(next_obs["action_mask"], dtype=np.float32),
                done,
            )
            obs = next_obs
            ep_return += reward
            global_step += 1

            if (
                len(replay) >= int(train_cfg["learning_starts"])
                and global_step % int(train_cfg["train_every"]) == 0
            ):
                b_obs, b_actions, b_rewards, b_next_obs, b_next_masks, b_dones = replay.sample(int(train_cfg["batch_size"]))
                obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(b_actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.as_tensor(b_rewards, dtype=torch.float32, device=device)
                next_obs_t = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
                next_masks_t = torch.as_tensor(b_next_masks, dtype=torch.bool, device=device)
                dones_t = torch.as_tensor(b_dones, dtype=torch.float32, device=device)

                q = q_net(obs_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    if bool(train_cfg["double_dqn"]):
                        online_next = q_net(next_obs_t).masked_fill(~next_masks_t, -1e9)
                        next_actions = online_next.argmax(dim=1, keepdim=True)
                        next_q = target_net(next_obs_t).gather(1, next_actions).squeeze(1)
                    else:
                        next_q = target_net(next_obs_t).masked_fill(~next_masks_t, -1e9).max(dim=1).values
                    target = rewards_t + float(train_cfg["gamma"]) * (1.0 - dones_t) * next_q

                loss = loss_fn(q, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                last_loss = float(loss.item())

            if global_step % int(train_cfg["target_update_every"]) == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        rows.append(
            {
                "episode": episode,
                "step": global_step,
                "episode_return": ep_return,
                "epsilon": epsilon_by_step(global_step, train_cfg),
                "loss": last_loss,
                "assign_actions": action_counts["assign"],
                "charge_actions": action_counts["charge"],
                "noop_actions": action_counts["noop"],
                "charge_open_steps": charge_open_steps,
            }
        )
        print(
            f"episode={episode} step={global_step} return={ep_return:.2f} "
            f"epsilon={rows[-1]['epsilon']:.3f} loss={last_loss} "
            f"charge={action_counts['charge']}/{charge_open_steps}"
        )

    with open(train_cfg["log_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "step",
                "episode_return",
                "epsilon",
                "loss",
                "assign_actions",
                "charge_actions",
                "noop_actions",
                "charge_open_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    save_checkpoint(train_cfg["weight_path"], env_cfg, q_net, train_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn.yaml")
    args = parser.parse_args()
    train(args.config)
