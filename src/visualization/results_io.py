#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
実験結果 JSON の読み込みと対貪欲比の計算（プロットスクリプト共通）
"""

import os
import glob
import json
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))


def setup_japanese_font():
    """日本語フォント設定（matplotlib_fontja があれば使用、なければフォールバック）"""
    try:
        import matplotlib_fontja  # noqa: F401
    except ImportError:
        import matplotlib.pyplot as plt
        for name in ["Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans",
                     "Yu Gothic", "Meiryo"]:
            try:
                plt.rcParams["font.family"] = name
                break
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False


def load_latest_result(env_tag: str, method_tag: str) -> dict:
    """
    指定環境・手法の最新の結果 JSON をロード

    Args:
        env_tag: 'static' または 'dynamic_churn{strength}'
        method_tag: 'greedy', 'proposed', 'surrogate_gamma{γ}'
    Returns:
        結果辞書（見つからなければ None）
    """
    pattern = os.path.join(PROJECT_ROOT, "results", env_tag,
                           f"{method_tag}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def match_ratio_vs_greedy(env_tag: str, method_tag: str):
    """
    試行ごとの対貪欲比（マッチ数）の平均と 95% 信頼区間を計算

    同一データ・同一シードで走らせた greedy の結果と試行番号で対応づける。

    Returns:
        (mean, ci95) のタプル。結果がなければ (None, None)
    """
    greedy = load_latest_result(env_tag, "greedy")
    target = load_latest_result(env_tag, method_tag)
    if greedy is None or target is None:
        return None, None

    g = np.array([t["match"] for t in greedy["trials"]], dtype=float)
    m = np.array([t["match"] for t in target["trials"]], dtype=float)
    n = min(len(g), len(m))
    g, m = g[:n], m[:n]
    valid = g > 0
    ratios = m[valid] / g[valid]
    mean = float(np.mean(ratios))
    ci95 = float(1.96 * np.std(ratios, ddof=1) / np.sqrt(len(ratios))) \
        if len(ratios) > 1 else 0.0
    return mean, ci95
