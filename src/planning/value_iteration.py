#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
価値反復（有限ホライズンの後ろ向き帰納法）による最適価値関数の計算（本研究のコア）

マッチ成立を吸収状態とする MDP の価値関数 V_h(F, R) を、
残りホライズン h = 0..EXPERIMENT_STEPS について計算する。
V_h(F, R) は「状態 (F, R) から最適方策に従ったとき、残り h ステップ以内に
マッチ成立へ至る確率」であり、目的関数（マッチ率）と厳密に一致する。

ベルマン方程式（TOP_K = 1、候補集合 J はステップごとに再抽選）:
    V_h(F, R) = E_J [ max_{j in J} { m_j
                  + (1 - c) (a_j - m_j) V_{h-1}(F+1, 0)
                  + (1 - c) (1 - a_j)  V_{h-1}(F, R+1) } ]
    a_j = P_user(j) * s(F, R)  （動的環境。静的環境では a_j = P_user(j)）
    m_j = a_j * P_comp(j)
    c   = P_churn(F, R)

候補集合の分布は JSAI 2026 版のデータ生成と同一
（P_user ~ U(0, 0.3)、P_comp ~ U(0, 0.3) * U(0, 0.3)）で、
期待値 E_J はモンテカルロで評価する（全状態で共通乱数を使用）。
共通乱数のため、静的環境では V が状態によらず厳密に一定となり、
ΔV = 0、すなわち提案方策は貪欲方策と厳密に一致する。
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.utils import calculate_user_state_score, selection_factor
from config import (
    ITEM_NUM,
    EXPERIMENT_STEPS,
    MAX_ACTION_COUNT,
    MAX_STEP_COUNT,
    VI_NUM_CANDIDATE_SETS,
    VI_CHUNK_SIZE,
    VI_CACHE_DIR,
    VI_SEED,
)


def state_score_grid() -> np.ndarray:
    """ユーザ状態スコア s(F, R) のグリッド (F: 0..100, R: 0..100)"""
    grid = np.zeros((MAX_ACTION_COUNT + 1, MAX_STEP_COUNT + 1))
    for f in range(MAX_ACTION_COUNT + 1):
        for r in range(MAX_STEP_COUNT + 1):
            grid[f, r] = calculate_user_state_score(f, r)
    return grid


def churn_grid(churn_strength: float, s_grid: np.ndarray) -> np.ndarray:
    """離反確率 P_churn(F, R) のグリッド"""
    return np.clip(churn_strength * (1.0 - s_grid / 2.0), 0.0, 1.0)


def sample_candidate_sets(num_sets: int, seed: int):
    """候補求人集合をサンプリング（JSAI 2026 版の生成分布と同一）"""
    rng = np.random.default_rng(seed)
    p_user = rng.uniform(0, 0.3, size=(num_sets, ITEM_NUM))
    p_comp = rng.uniform(0, 0.3, size=(num_sets, ITEM_NUM)) * rng.uniform(
        0, 0.3, size=(num_sets, ITEM_NUM)
    )
    return p_user, p_comp


def compute_value_tables(
    decay_flag: bool,
    churn_strength: float,
    horizon: int = EXPERIMENT_STEPS,
    num_sets: int = VI_NUM_CANDIDATE_SETS,
    seed: int = VI_SEED,
    selection_strength: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    後ろ向き帰納法で V を計算

    Returns:
        V: shape (horizon + 1, 101, 101) の float32 配列。
           V[h, F, R] = 残り h ステップでのマッチ確率（V[0] = 0）
    """
    s_grid = state_score_grid()                      # (101, 101)
    c_grid = churn_grid(churn_strength, s_grid)      # (101, 101)
    mult = s_grid if decay_flag else np.ones_like(s_grid)
    # 選抜効果 h(F)（selection_strength = 0 なら全て 1）
    h_grid = np.array([selection_factor(f, selection_strength)
                       for f in range(MAX_ACTION_COUNT + 1)])

    p_user, p_comp = sample_candidate_sets(num_sets, seed)

    n_f, n_r = s_grid.shape
    n_states = n_f * n_r
    mult_flat = mult.reshape(-1)
    c_flat = c_grid.reshape(-1)

    # 遷移先インデックス（上限でクリップ、JSAI 2026 版の状態上限と同一）
    f_idx, r_idx = np.meshgrid(np.arange(n_f), np.arange(n_r), indexing="ij")
    f_next = np.minimum(f_idx + 1, MAX_ACTION_COUNT)   # 行動あり → (F+1, 0)
    r_next = np.minimum(r_idx + 1, MAX_STEP_COUNT)     # 行動なし → (F, R+1)

    V = np.zeros((horizon + 1, n_f, n_r), dtype=np.float32)

    for h in range(1, horizon + 1):
        v_prev = V[h - 1].astype(np.float64)
        va = v_prev[f_next, 0].reshape(-1)     # V_{h-1}(F+1, 0)
        vr = v_prev[f_idx, r_next].reshape(-1) # V_{h-1}(F, R+1)

        # スコアの分解（j に依存しない定数項を分離）:
        #   score_j = (1-c) vr + a_j * [ pc_j (1 - (1-c) va) + (1-c)(va - vr) ]
        # ここで a_j = mult * pu_j なので、状態ごとのスカラー
        #   A = 1 - (1-c) va,  B = (1-c)(va - vr)
        # を用いて max_j pu_j (pc_j A + B) を評価すればよい
        h_flat = h_grid[f_idx].reshape(-1)      # 現在状態の F に対する選抜効果
        A = h_flat * (1.0 - (1.0 - c_flat) * va)   # (n_states,)
        B = (1.0 - c_flat) * (va - vr)         # (n_states,)

        g = np.zeros(n_states)
        for start in range(0, num_sets, VI_CHUNK_SIZE):
            pu = p_user[start:start + VI_CHUNK_SIZE]   # (chunk, ITEM_NUM)
            pc = p_comp[start:start + VI_CHUNK_SIZE]
            # (n_states, chunk, ITEM_NUM)
            vals = pu[None, :, :] * (
                pc[None, :, :] * A[:, None, None] + B[:, None, None]
            )
            g += vals.max(axis=2).sum(axis=1)
        g /= num_sets

        v_new = (1.0 - c_flat) * vr + mult_flat * g
        V[h] = np.clip(v_new, 0.0, 1.0).reshape(n_f, n_r).astype(np.float32)

        if verbose and h % 10 == 0:
            print(f"  価値反復: 残りホライズン {h}/{horizon}, "
                  f"V[初期状態(0,0)] = {V[h][0, 0]:.4f}")

    return V


def value_table_path(decay_flag: bool, churn_strength: float,
                     selection_strength: float = 0.0,
                     cache_dir: str = None) -> str:
    """価値関数テーブルのキャッシュファイルパス"""
    if cache_dir is None:
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        cache_dir = os.path.join(project_root, VI_CACHE_DIR)
    env = "dynamic" if decay_flag else "static"
    sel = f"_sel{selection_strength}" if selection_strength > 0 else ""
    return os.path.join(cache_dir, f"V_{env}_churn{churn_strength}{sel}.npz")


def get_or_compute_value_tables(decay_flag: bool, churn_strength: float,
                                selection_strength: float = 0.0,
                                verbose: bool = True) -> str:
    """キャッシュがあればそれを使い、なければ計算して保存。パスを返す"""
    path = value_table_path(decay_flag, churn_strength, selection_strength)
    if os.path.exists(path):
        if verbose:
            print(f"価値関数テーブルをキャッシュからロード: {path}")
        return path
    if verbose:
        print(f"価値関数テーブルを計算中 (decay={decay_flag}, "
              f"churn={churn_strength}, selection={selection_strength})...")
    V = compute_value_tables(decay_flag, churn_strength,
                             selection_strength=selection_strength,
                             verbose=verbose)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, V=V)
    if verbose:
        print(f"価値関数テーブルを保存: {path}")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="価値関数テーブルを事前計算")
    parser.add_argument("--churn_strength", type=float, default=0.0)
    parser.add_argument("--no_decay_flag", action="store_false",
                        dest="decay_flag")
    parser.add_argument("--selection_strength", type=float, default=0.0)
    args = parser.parse_args()
    get_or_compute_value_tables(args.decay_flag, args.churn_strength,
                                args.selection_strength)


if __name__ == "__main__":
    main()
