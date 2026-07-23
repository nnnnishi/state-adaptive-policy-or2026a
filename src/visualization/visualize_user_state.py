#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
実験設定スライド用の環境可視化（3D サーフェス 2 枚）

- 求職者行動スコア s(F, R)（動的環境で P_user に乗算）
- 継続確率 1 - P_churn(F, R)（スコアの重み (1 - P_churn) ΔV(s) に対応）

両者は同じ単調性（F で増加・R で減少）を持つため、同一の視点
（elev=30, azim=-30, 両軸反転）で並べて表示できる。
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401  日本語フォント設定

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.utils import calculate_user_state_score


def _surface(Z, title, output_path):
    F = np.arange(0, 101, 2)
    R = np.arange(0, 101, 2)
    FF, RR = np.meshgrid(F, R, indexing="ij")
    fig = plt.figure(figsize=(4.8, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(FF, RR, Z, cmap="viridis", rstride=1, cstride=1,
                    linewidth=0, antialiased=True)
    ax.set_xlabel("行動数 F", fontsize=11, labelpad=8)
    ax.set_ylabel("経過ステップ R", fontsize=11, labelpad=8)
    ax.view_init(elev=30, azim=-30)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(labelsize=8)
    fig.savefig(output_path, dpi=200, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--churn_strength", type=float, default=0.1,
                        help="継続確率の図に用いる離反の強さ")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    F = np.arange(0, 101, 2)
    R = np.arange(0, 101, 2)
    FF, RR = np.meshgrid(F, R, indexing="ij")
    S = np.vectorize(calculate_user_state_score)(FF, RR)
    SURV = 1.0 - args.churn_strength * (1.0 - S / 2.0)

    _surface(S, "求職者行動スコア s(F, R)",
             os.path.join(output_dir, "env_score.png"))
    _surface(SURV, "継続確率 1−P_churn(F, R)",
             os.path.join(output_dir, "env_continuation.png"))


if __name__ == "__main__":
    main()
