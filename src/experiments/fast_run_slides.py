# -*- coding: utf-8 -*-
"""高速ベクトル化ランナー（v3定式化: 離脱のみ吸収、行動時点でマッチ期待計上）"""
import os
import sys
import json
import time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from utils.utils import calculate_user_state_score
from planning.value_iteration import get_or_compute_value_tables
from config import (USER_NUM, ITEM_NUM, EXPERIMENT_STEPS, RANDOM_SEED,
                    MAX_ACTION_COUNT, MAX_STEP_COUNT)

S_GRID = np.array([[calculate_user_state_score(f, r)
                    for r in range(MAX_STEP_COUNT + 1)]
                   for f in range(MAX_ACTION_COUNT + 1)])
DELTA_GRID = S_GRID[np.minimum(np.arange(101) + 1, 100), 0][:, None] - S_GRID
_VCACHE = {}


def vtable(decay_flag, churn):
    key = (decay_flag, churn)
    if key not in _VCACHE:
        path = get_or_compute_value_tables(decay_flag, churn, verbose=False)
        _VCACHE[key] = np.load(path)["V"].astype(np.float64)
    return _VCACHE[key]


def gen_trial_data(trial):
    np.random.seed(RANDOM_SEED + trial)
    F0 = np.random.randint(0, 51, size=USER_NUM)
    R0 = np.random.randint(0, 51, size=USER_NUM)
    np.random.seed(RANDOM_SEED + trial)
    d = np.random.uniform(0, 0.3, size=(USER_NUM, EXPERIMENT_STEPS, ITEM_NUM, 3))
    return F0, R0, d[..., 0], d[..., 1] * d[..., 2]


def run_trial(trial, method, gamma, decay_flag, churn, V=None):
    F0, R0, p_user, p_comp = gen_trial_data(trial)
    rng = np.random.default_rng(RANDOM_SEED * 1000 + trial)
    U = rng.uniform(size=(EXPERIMENT_STEPS, USER_NUM, 3))

    F = F0.copy(); R = R0.copy()
    active = np.ones(USER_NUM, dtype=bool)
    n_action = n_match = n_churn = 0

    for step in range(EXPERIMENT_STEPS):
        if not active.any():
            break
        idx = np.where(active)[0]
        f = np.minimum(F[idx], MAX_ACTION_COUNT)
        r = np.minimum(R[idx], MAX_STEP_COUNT)
        s = S_GRID[f, r]
        c = np.clip(churn * (1.0 - s / 2.0), 0, 1) if churn > 0 else np.zeros_like(s)
        pu = p_user[idx, step]; pc = p_comp[idx, step]

        if method == "greedy":
            score = pu * pc
        elif method == "surrogate":
            delta = DELTA_GRID[f, r][:, None]
            score = pu * pc + gamma * pu * delta
        elif method == "proposed":
            remaining = EXPERIMENT_STEPS - step - 1
            v_next = V[remaining]
            va = v_next[np.minimum(f + 1, MAX_ACTION_COUNT), 0][:, None]
            vr = v_next[f, np.minimum(r + 1, MAX_STEP_COUNT)][:, None]
            mult = s[:, None] if decay_flag else 1.0
            a = np.minimum(pu * mult, 1.0)
            cc = c[:, None]
            # v3: 報酬（a*pc）は遷移と独立に計上
            score = a * pc + (1 - cc) * (a * va + (1 - a) * vr)
        else:
            raise ValueError(method)

        j = np.argmax(score, axis=1)
        pu_j = pu[np.arange(len(idx)), j]
        pc_j = pc[np.arange(len(idx)), j]
        a_actual = pu_j * s if decay_flag else pu_j

        act = U[step, idx, 0] < a_actual
        match = act & (U[step, idx, 1] < pc_j)   # v3: 計上のみ、除外しない
        churned = U[step, idx, 2] < c

        n_action += int(act.sum())
        n_match += int(match.sum())
        n_churn += int(churned.sum())

        F[idx] = np.where(act, F[idx] + 1, F[idx])
        R[idx] = np.where(act, 0, R[idx] + 1)
        active[idx[churned]] = False

    return {"user_action": n_action, "match": n_match, "churned": n_churn}


def run_config(method, gamma, decay_flag, churn, n_trials=100):
    V = vtable(decay_flag, churn) if method == "proposed" else None
    t0 = time.time()
    trials = [run_trial(t, method, gamma, decay_flag, churn, V)
              for t in range(n_trials)]
    res = {
        "method": method, "gamma": gamma, "decay": decay_flag,
        "churn_strength": churn, "trials": trials,
        "average": {k: float(np.mean([t[k] for t in trials]))
                    for k in ["user_action", "match", "churned"]},
        "execution_time": time.time() - t0,
        "runner": "fast_vectorized_crn_v3",
    }
    env = "static" if not decay_flag else f"dynamic_churn{churn}"
    tag = f"surrogate_gamma{gamma}" if method == "surrogate" else method
    outdir = os.path.join(REPO, "results", env)
    os.makedirs(outdir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(outdir, f"{tag}_{ts}.json"), "w") as fj:
        json.dump(res, fj)
    print(f"{env:22s} {tag:22s} match={res['average']['match']:8.1f} "
          f"({res['execution_time']:.1f}s)")
    return res


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("configs")
    args = p.parse_args()
    for cfg in args.configs.split(";"):
        parts = cfg.strip().split(",")
        run_config(parts[0], float(parts[1]), parts[2] == "1", float(parts[3]))
