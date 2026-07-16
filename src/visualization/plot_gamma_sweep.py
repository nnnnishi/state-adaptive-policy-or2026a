#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
γ（先行研究の将来効果重み）感度プロット（発表スライド右図）

固定した環境（デフォルト: 動的 + 離反 0.05）で先行研究の γ を掃引し、
γ を持たない提案手法を水平線として重ねる。
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import GAMMA_GRID
from visualization.results_io import (
    PROJECT_ROOT, setup_japanese_font, match_ratio_vs_greedy,
)

setup_japanese_font()
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
})

PROPOSED_COLOR = "#0E6080"
SURROGATE_COLOR = "#6E7B85"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--churn_strength", type=float, default=0.05,
                        help="対象環境の離反の強さ")
    parser.add_argument("--gammas", type=float, nargs="*", default=GAMMA_GRID)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    env = f"dynamic_churn{args.churn_strength}"

    xs, means, cis = [], [], []
    for gamma in args.gammas:
        mean, ci = match_ratio_vs_greedy(env, f"surrogate_gamma{gamma}")
        if mean is None:
            print(f"警告: {env} surrogate γ={gamma} の結果が見つかりません")
            continue
        xs.append(gamma)
        means.append(mean)
        cis.append(ci)

    prop_mean, prop_ci = match_ratio_vs_greedy(env, "proposed")

    if not xs and prop_mean is None:
        print("エラー: プロット可能な結果がありません。先に実験を実行してください。")
        return 1

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhline(1.0, color="#808080", lw=1.2, ls=":")

    if prop_mean is not None:
        ax.axhline(prop_mean, color=PROPOSED_COLOR, lw=2.5,
                   label="提案手法（γ 不要）")
        ax.axhspan(prop_mean - prop_ci, prop_mean + prop_ci,
                   color=PROPOSED_COLOR, alpha=0.15)

    if xs:
        ax.errorbar(xs, means, yerr=cis, fmt="o--", color=SURROGATE_COLOR,
                    lw=2, ms=7, capsize=3, label="先行研究 [西村+ 2025]")
        ax.set_xscale("log")

    ax.set_xlabel("将来効果の重視度 γ（先行研究のパラメータ）")
    ax.set_ylabel("マッチ数の対貪欲比")
    ax.set_title(f"動的環境・離反の強さ {args.churn_strength} での γ 感度")
    ax.legend(loc="best", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    output = args.output or os.path.join(
        PROJECT_ROOT, "results", f"gamma_sweep_churn{args.churn_strength}.pdf")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.replace(".pdf", ".png"), dpi=200)
    print(f"保存: {output}")
    return 0


if __name__ == "__main__":
    exit(main())
