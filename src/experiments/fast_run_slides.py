# -*- coding: utf-8 -*-
"""
高速ベクトル化ランナー（スライド更新用）

リポジトリ state-adaptive-policy-or2026a と同一のモデル・パラメータで実験を実行する。
- 人工データ: create_data.py と同一シード・同一乱数列を numpy で厳密に再現
  （ユーザ初期状態・求人確率は CSV 生成版とビット単位で一致）
- 方策・遷移・離反: experiment.py と同一のロジックをベクトル化
- ベルヌーイ判定の乱数列のみ共通乱数（CRN）方式に変更:
  (trial, user, step) ごとに事前に u_act, u_match, u_churn を引くため、
  同一環境では手法間の比較が対応比較（分散低減）になる
結果 JSON はリポジトリと同一フォーマットで results/ に保存する。
"""
import os
import sys
import json
import time
import numpy as np

REPO = "/tmp/exprepo"
sys.path.insert(0, os.path.join(REPO, "src"))

from utils.utils import calculate_user_state_score  # noqa: E402
from planning.value_iteration import get_or_compute_value_tables  # noqa: E402
from config import (  # noqa: E402
    USER_NUM, ITEM_NUM, EXPERIMENT_STEPS, RANDOM_SEED,
    MAX_ACTION_COUNT, MAX_STEP_COUNT,
)

# 状態スコアと Δ(s) のルックアップテーブル
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
    """create_data.py と同一の乱数列でユーザ初期状態と求人確率を再現"""
    np.random.seed(RANDOM_SEED + trial)
    F0 = np.random.randint(0, 51, size=USER_NUM)
    R0 = np.random.randint(0, 51, size=USER_NUM)
    np.random.seed(RANDOM_SEED + trial)
    d = np.random.uniform(0, 0.3, size=(USER_NUM, EXPERIMENT_STEPS, ITEM_NUM, 3))
    p_user = d[..., 0]
    p_comp = d[..., 1] * d[..., 2]
    return F0, R0, p_user, p_comp


def run_trial(trial, method, gamma, decay_flag, churn, V=None):
    F0, R0, p_user, p_comp = gen_trial_data(trial)
    rng = np.random.default_rng(RANDOM_SEED * 1000 + trial)
    U = rng.uniform(size=(EXPERIMENT_STEPS, USER_NUM, 3))  # act, match, churn

    F = F0.copy()
    R = R0.copy()
    active = np.ones(USER_NUM, dtype=bool)
    n_action = 0
    n_match = 0
    n_churn = 0

    for step in range(EXPERIMENT_STEPS):
        if not active.any():
            break
        idx = np.where(active)[0]
        f = np.minimum(F[idx], MAX_ACTION_COUNT)
        r = np.minimum(R[idx], MAX_STEP_COUNT)
        s = S_GRID[f, r]
        c = np.clip(churn * (1.0 - s / 2.0), 0, 1) if churn > 0 else np.zeros_like(s)
        pu = p_user[idx, step]   # (n, ITEM)
        pc = p_comp[idx, step]

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
            m = a * pc
            cc = c[:, None]
            score = m + (1 - cc) * (a - m) * va + (1 - cc) * (1 - a) * vr
        else:
            raise ValueError(method)

        j = np.argmax(score, axis=1)
        pu_j = pu[np.arange(len(idx)), j]
        pc_j = pc[np.arange(len(idx)), j]
        a_actual = pu_j * s if decay_flag else pu_j

        act = U[step, idx, 0] < a_actual
        match = act & (U[step, idx, 1] < pc_j)
        churned = (~match) & (U[step, idx, 2] < c)

        n_action += int(act.sum())
        n_match += int(match.sum())
        n_churn += int(churned.sum())

        F[idx] = np.where(act, F[idx] + 1, F[idx])
        R[idx] = np.where(act, 0, R[idx] + 1)
        active[idx[match | churned]] = False

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
        "runner": "fast_vectorized_crn",
    }
    env = "static" if not decay_flag else f"dynamic_churn{churn}"
    tag = f"surrogate_gamma{gamma}" if method == "surrogate" else method
    outdir = os.path.join(REPO, "results", env)
    os.makedirs(outdir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outdir, f"{tag}_{ts}.json")
    with open(path, "w") as fjson:
        json.dump(res, fjson)
    print(f"{env:22s} {tag:22s} match={res['average']['match']:8.1f} "
          f"({res['execution_time']:.1f}s)")
    return res


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("configs", help="例: 'greedy:0:static 0.1:0.05' 等は呼び出し側で構成")
    args = p.parse_args()
    for cfg in args.configs.split(";"):
        parts = cfg.strip().split(",")
        method, gamma, decay, churn = (
            parts[0], float(parts[1]), parts[2] == "1", float(parts[3]))
        run_config(method, gamma, decay, churn)
