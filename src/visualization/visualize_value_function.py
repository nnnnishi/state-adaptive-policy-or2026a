#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
価値関数 V(F, R) と価値増分 ΔV(F, R) の可視化（発表スライドの解釈図）

- V の 3D サーフェス
- ΔV(F, R) = V(F+1, 0) - V(F, R+1) のヒートマップ
  （序盤・離反期で大、定着期で小、というキャリアアドバイザの経験則と対応）
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

from config import EXPERIMENT_STEPS, MAX_ACTION_COUNT, MAX_STEP_COUNT
from planning.value_iteration import get_or_compute_value_tables
from visualization.results_io import PROJECT_ROOT, setup_japanese_font

setup_japanese_font()
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--churn_strength", type=float, default=0.05)
    parser.add_argument("--no_decay_flag", action="store_false",
                        dest="decay_flag")
    parser.add_argument("--selection_strength", type=float, default=0.0)
    parser.add_argument("--max_f", type=int, default=50,
                        help="表示する F の上限")
    parser.add_argument("--max_r", type=int, default=50,
                        help="表示する R の上限")
    args = parser.parse_args()

    path = get_or_compute_value_tables(args.decay_flag, args.churn_strength,
                                       args.selection_strength)
    V = np.load(path)["V"][EXPERIMENT_STEPS]  # フルホライズンの価値関数

    nf, nr = args.max_f + 1, args.max_r + 1
    Vv = V[:nf, :nr].astype(float)

    # ΔV(F, R) = V(F+1, 0) - V(F, R+1)
    f_idx = np.arange(nf)
    r_idx = np.arange(nr)
    va = V[np.minimum(f_idx + 1, MAX_ACTION_COUNT), 0][:, None]
    vr = V[f_idx[:, None], np.minimum(r_idx + 1, MAX_STEP_COUNT)[None, :]]
    dV = (va - vr).astype(float)

    env = "dynamic" if args.decay_flag else "static"
    sel = f"_sel{args.selection_strength}" if args.selection_strength > 0 else ""
    suffix = f"{env}_churn{args.churn_strength}{sel}"

    # ---- V の 3D サーフェス ----
    FF, RR = np.meshgrid(f_idx, r_idx, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(RR, FF, Vv, cmap="viridis", linewidth=0)
    ax.set_xlabel("最終行動からの経過ステップ数 R")
    ax.set_ylabel("累積行動数 F")
    ax.set_zlabel("V(F, R)")
    ax.set_title("最適価値関数 V（計画期間内のマッチ確率）")
    out1 = os.path.join(PROJECT_ROOT, "results",
                        f"value_function_3d_{suffix}.pdf")
    fig.savefig(out1, bbox_inches="tight")
    fig.savefig(out1.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out1}")

    # ---- ΔV のヒートマップ ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(dV, origin="lower", aspect="auto", cmap="viridis")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("ΔV(F, R)")
    ax.set_xlabel("最終行動からの経過ステップ数 R")
    ax.set_ylabel("累積行動数 F")
    ax.set_title("行動による価値増分 ΔV = V(F+1, 0) − V(F, R+1)")
    out2 = os.path.join(PROJECT_ROOT, "results",
                        f"delta_v_heatmap_{suffix}.pdf")
    fig.tight_layout()
    fig.savefig(out2)
    fig.savefig(out2.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"保存: {out2}")


if __name__ == "__main__":
    main()
