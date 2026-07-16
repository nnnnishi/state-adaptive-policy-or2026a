#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
単調性制約による価値関数の平滑化（論文 3.2 節）

観測データから推定した価値関数はスパースな状態でノイズが大きいため、
ドメイン知識として期待される単調性
  - F（累積行動数）について単調増加
  - R（経過ステップ数）について単調減少
を課した重み付き最小二乗（凸二次計画）で平滑化する:

  min_x  sum_{f,r} w_{f,r} (x_{f,r} - xhat_{f,r})^2
  s.t.   x_{f+1,r} >= x_{f,r},   x_{f,r+1} <= x_{f,r}

実装は Dykstra の交互射影法による（各行・各列への等張回帰
（PAVA: Pool Adjacent Violators Algorithm）の繰り返し）。
本数値実験（真の確率が既知）では価値反復が厳密なので平滑化は不要だが、
実データで V を推定する場合のリファレンス実装として同梱する。
"""

import numpy as np


def _pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """重み付き等張回帰（単調増加）: Pool Adjacent Violators Algorithm"""
    n = len(y)
    # ブロックのスタック（値・重み・サイズ）
    vals, wts, sizes = [], [], []
    for i in range(n):
        vals.append(float(y[i]))
        wts.append(float(w[i]))
        sizes.append(1)
        # 単調性が破れている限り隣接ブロックをマージ
        while len(vals) > 1 and vals[-2] > vals[-1]:
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (
                wts[-2] + wts[-1])
            wts[-2] += wts[-1]
            sizes[-2] += sizes[-1]
            vals[-2] = merged
            vals.pop()
            wts.pop()
            sizes.pop()
    out = np.empty(n)
    pos = 0
    for v, s in zip(vals, sizes):
        out[pos:pos + s] = v
        pos += s
    return out


def _isotonic_rows_increasing(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """各列について F 方向（行方向）に単調増加へ射影"""
    out = x.copy()
    for r in range(x.shape[1]):
        out[:, r] = _pava_increasing(x[:, r], w[:, r])
    return out


def _isotonic_cols_decreasing(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """各行について R 方向（列方向）に単調減少へ射影（符号反転して増加射影）"""
    out = x.copy()
    for f in range(x.shape[0]):
        out[f, :] = -_pava_increasing(-x[f, :], w[f, :])
    return out


def smooth_value_table(
    v_hat: np.ndarray,
    weights: np.ndarray = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    単調性制約付きで価値関数テーブルを平滑化

    Args:
        v_hat: 推定された価値関数 (n_F, n_R)
        weights: 重み（例: 状態ごとのサンプルサイズ）。None なら等重み
        max_iter: Dykstra 反復の最大回数
        tol: 収束判定閾値

    Returns:
        v: 単調性（F 増加・R 減少）を満たす平滑化済みテーブル
    """
    if weights is None:
        weights = np.ones_like(v_hat)
    weights = np.maximum(weights.astype(float), 1e-12)

    x = v_hat.astype(float).copy()
    p = np.zeros_like(x)  # 行制約の補正項
    q = np.zeros_like(x)  # 列制約の補正項

    for it in range(max_iter):
        x_prev = x.copy()
        y = _isotonic_rows_increasing(x + p, weights)
        p = (x + p) - y
        x = _isotonic_cols_decreasing(y + q, weights)
        q = (y + q) - x
        if np.max(np.abs(x - x_prev)) < tol:
            break

    return x


def main():
    """簡単な動作デモ: ノイズ入りの単調テーブルを復元"""
    rng = np.random.default_rng(0)
    n_f, n_r = 30, 30
    f = np.arange(n_f)[:, None]
    r = np.arange(n_r)[None, :]
    true_v = (1 - np.exp(-(f + 1) / 8.0)) * np.exp(-r / 10.0) * 0.6
    sample_size = rng.integers(1, 200, size=(n_f, n_r))
    noise = rng.normal(0, 0.15 / np.sqrt(sample_size))
    v_hat = np.clip(true_v + noise, 0, 1)

    v_smooth = smooth_value_table(v_hat, weights=sample_size)

    rmse_before = np.sqrt(np.mean((v_hat - true_v) ** 2))
    rmse_after = np.sqrt(np.mean((v_smooth - true_v) ** 2))
    print(f"RMSE 平滑化前: {rmse_before:.4f} → 平滑化後: {rmse_after:.4f}")

    # 単調性の検証
    assert np.all(np.diff(v_smooth, axis=0) >= -1e-6), "F 方向の単調増加が破れています"
    assert np.all(np.diff(v_smooth, axis=1) <= 1e-6), "R 方向の単調減少が破れています"
    print("単調性制約を満たすことを確認しました")


if __name__ == "__main__":
    main()
