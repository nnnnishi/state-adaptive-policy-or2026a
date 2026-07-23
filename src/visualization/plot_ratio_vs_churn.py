#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
離反の強さ × マッチング数の対貪欲比プロット（発表スライド左図）

- 状態適応的方策（γ なし）1 本
- 代理目的関数の γ 固定線を複数本
- x = 0 は静的環境（状態適応的方策は貪欲と厳密に一致し、比率 1.0）
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

from config import CHURN_GRID
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
GAMMA_STYLES = [
    ("^:", "#B8BEC4"),
    ("s--", "#8A96A0"),
    ("D-.", "#5F6B75"),
    ("v:", "#B57B7B"),
]


def collect_series(method_tag, churn_grid):
    """静的環境 + 動的環境の churn グリッドに沿って対貪欲比を収集"""
    xs, means, cis = [], [], []
    # x=0: 静的環境
    mean, ci = match_ratio_vs_greedy("static", method_tag)
    if mean is not None:
        xs.append(0.0)
        means.append(mean)
        cis.append(ci)
    # x>0: 動的 + 離反環境
    for c in churn_grid:
        if c <= 0:
            continue
        mean, ci = match_ratio_vs_greedy(f"dynamic_churn{c}", method_tag)
        if mean is not None:
            xs.append(c)
            means.append(mean)
            cis.append(ci)
    return np.array(xs), np.array(means), np.array(cis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", type=float, nargs="*",
                        default=[0.1, 1.0, 10.0],
                        help="プロットする先行研究の γ 値")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhline(1.0, color="#808080", lw=1.2, ls=":", zorder=1)

    plotted_any = False
    for (style, color), gamma in zip(GAMMA_STYLES, args.gammas):
        xs, means, cis = collect_series(f"surrogate_gamma{gamma}", CHURN_GRID)
        if len(xs) == 0:
            print(f"警告: surrogate γ={gamma} の結果が見つかりません")
            continue
        ax.errorbar(xs, means, yerr=cis, fmt=style, color=color, lw=1.8,
                    ms=6, capsize=3, label=f"代理目的関数 γ={gamma}", zorder=2)
        plotted_any = True

    xs, means, cis = collect_series("proposed", CHURN_GRID)
    if len(xs) > 0:
        ax.errorbar(xs, means, yerr=cis, fmt="o-", color=PROPOSED_COLOR,
                    lw=2.5, ms=8, capsize=3, label="状態適応的方策（調整パラメータなし）",
                    zorder=5)
        plotted_any = True

    if not plotted_any:
        print("エラー: プロット可能な結果がありません。先に実験を実行してください。")
        return 1

    ax.set_xlabel("離反の強さ（0 = 離反なし・静的環境）")
    ax.set_ylabel("マッチング数の対貪欲比")
    ax.legend(loc="best", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    output = args.output or os.path.join(
        PROJECT_ROOT, "results", "ratio_vs_churn.pdf")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.replace(".pdf", ".png"), dpi=200)
    print(f"保存: {output}")
    return 0


if __name__ == "__main__":
    exit(main())
