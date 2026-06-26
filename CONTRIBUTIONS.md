# Contribution and integration note

The Git history uses the `sezenbalkan` account as the repository integration
account. Commit authorship therefore does not by itself identify individual
ownership. Individual grading ownership is:

- **Sezen Balkan — Role A:** `code/dqn_agent.py`, `code/train_dqn.py`,
  `code/train_factored_dqn.py`, Role-A configs, weights, logs, and report text.
- **Ozan Karhan — Role B:** `code/role_b/`, Role-B configs, weights, logs,
  figures, and report text.
- **Tuba Nur Büyükata — Role C:** `code/role_c/`,
  `configs/role_c_rollout.yaml`, planner settings/logs, and report text.
- **Joint:** offline RL, multi-agent IDQN, pooled dataset, `run_all.py`, and
  final report integration.

The final cleanup, reproducibility audit, and shared-table integration were
committed through the integration account. Each owner remains responsible for
explaining and modifying their method during the oral defense.
