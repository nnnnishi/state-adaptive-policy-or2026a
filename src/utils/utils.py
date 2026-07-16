"""
2段階モデル用ユーティリティ関数
"""

import numpy as np
from typing import List
import os
import sys

from models.models import User, Item

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class UserStateScoreParams:
    """ユーザ状態スコア計算用パラメータ

    Attributes:
        MAX_ACTION_COUNT (int): 行動数の上限。これを超えると成長が停止
        MAX_STEPS (int): 経過ステップの上限。これを超えると減衰が停止
        BASE_SCORE (float): 初期スコア。(action_count, steps_since_last_action) = (0, 0) の時の値
        MAX_SCORE_MULTIPLIER (float): 最大スコア倍率。action_count = MAX_ACTION_COUNT の時のスコア
        DECAY_RATE (float): 経過ステップあたりの減衰率。
                           MAX_STEPS ステップ後にスコアが0.05になるよう設定
    """

    MAX_ACTION_COUNT = 100
    MAX_STEPS = 100
    BASE_SCORE = 1.0
    MAX_SCORE_MULTIPLIER = 3.0
    DECAY_RATE = 0.05 ** (1 / MAX_STEPS)


def calculate_user_state_score(action_count: int, steps_since_last_action: int) -> float:
    """
    ユーザ状態スコアを計算

    Args:
        action_count: 累積オンライン行動数（最大100）
        steps_since_last_action: 最終行動からの経過ステップ数（最大100）

    Returns:
        user_state_score: [0,2] の範囲のスコア。初期値は (0, 0) で 1
    """
    # 上限値に制限
    action_count = min(action_count, UserStateScoreParams.MAX_ACTION_COUNT)
    steps_since_last_action = min(steps_since_last_action, UserStateScoreParams.MAX_STEPS)

    # 累積行動数の影響（対数的成長）
    if action_count == 0:
        score = UserStateScoreParams.BASE_SCORE
    else:
        score = UserStateScoreParams.BASE_SCORE + (
            np.log1p(action_count) / np.log1p(UserStateScoreParams.MAX_ACTION_COUNT)
        )

    # 経過ステップの影響（指数的減衰）
    score = score * (UserStateScoreParams.DECAY_RATE ** steps_since_last_action)

    return score


def get_user_state_score_delta(current_action_count: int, current_steps: int) -> float:
    """
    オンライン行動実行後のuser_state_scoreの変化量を計算

    Args:
        current_action_count: 現在の累積行動数
        current_steps: 現在の経過ステップ数
    Returns:
        delta: スコアの変化量
    """
    current_score = calculate_user_state_score(current_action_count, current_steps)
    next_score = calculate_user_state_score(current_action_count + 1, 0)
    return next_score - current_score


def calculate_step_ratios(history: dict, steps: list = [10, 20, 30, 40, 50]) -> dict:
    """
    特定ステップでのベースラインに対する比率を計算

    Args:
        history: ステップごとのスコア履歴
        steps: 比率を計算するステップのリスト

    Returns:
        step_ratios: 各ステップでの比率
    """
    step_ratios = {step: {"user_action": 0.0, "match": 0.0} for step in steps}

    for step in steps:
        if step >= len(history["baseline"]["user_action"]):
            continue

        # 各ステップまでの累積値を計算
        baseline_cumsum = {
            phase: sum(history["baseline"][phase][:step])
            for phase in ["user_action", "match"]
        }
        proposed_cumsum = {
            phase: sum(history["proposed"][phase][:step])
            for phase in ["user_action", "match"]
        }

        # 比率を計算
        for phase in ["user_action", "match"]:
            if baseline_cumsum[phase] > 0:
                step_ratios[step][phase] = (
                    proposed_cumsum[phase] / baseline_cumsum[phase]
                )
            else:
                step_ratios[step][phase] = 0.0

    return step_ratios


def calculate_metrics(
    baseline_score: dict, proposed_score: dict, history: dict = None
) -> dict:
    """
    ベースラインと提案手法の比較指標を計算

    Args:
        baseline_score: 各フェーズのベースラインスコア
        proposed_score: 各フェーズの提案手法スコア
        history: ステップごとのスコア履歴（オプション）

    Returns:
        metrics: 各種評価指標
    """
    metrics = {
        "raw_counts": {
            "baseline": baseline_score,
            "proposed": proposed_score,
        },
        "ratios": {},
        "step_ratios": {},
    }

    # ベースラインに対する比率を計算
    for phase in ["user_action", "match"]:
        if baseline_score[phase] > 0:
            metrics["ratios"][phase] = proposed_score[phase] / baseline_score[phase]
        else:
            metrics["ratios"][phase] = 0

    # 特定ステップでの比率を計算
    if history:
        metrics["step_ratios"] = calculate_step_ratios(history)

    return metrics


def churn_probability(action_count: int, steps_since_last_action: int,
                      churn_strength: float) -> float:
    """
    ユーザ状態に依存する離反確率を計算（本研究で追加）

    P_churn(F, R) = churn_strength * (1 - s(F, R) / 2)

    ユーザ状態スコア s は [0, 2] の範囲（JSAI 2026 版と同一の関数）を取り、
    活動的なユーザ（s が大きい）ほど離反しにくく、
    行動が止まったユーザ（s が小さい）ほど離反しやすい。

    Args:
        action_count: 累積行動数（Frequency）
        steps_since_last_action: 最終行動からの経過ステップ数（Recency）
        churn_strength: 離反の強さ（0 なら離反なし）
    Returns:
        churn_prob: [0, 1] の離反確率
    """
    if churn_strength <= 0:
        return 0.0
    score = calculate_user_state_score(action_count, steps_since_last_action)
    return float(np.clip(churn_strength * (1.0 - score / 2.0), 0.0, 1.0))


def selection_factor(action_count: int, selection_strength: float) -> float:
    """
    選抜効果（高望み）係数を計算（本研究で追加、感度分析用）

    h(F) = 1 / (1 + selection_strength * F)

    「応募数が多いのに未マッチ＝マッチしにくい」という負のシグナルを
    企業受容確率に乗算する: P_comp_eff = P_comp * h(F)。
    selection_strength = 0 で無効（デフォルト、JSAI 2026 版と同一の環境）。
    """
    if selection_strength <= 0:
        return 1.0
    f = min(action_count, UserStateScoreParams.MAX_ACTION_COUNT)
    return 1.0 / (1.0 + selection_strength * f)
