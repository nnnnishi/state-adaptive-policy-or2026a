#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
状態適応的方策による推薦システム評価の実験スクリプト（OR2026a）

シミュレータの骨格（データロード・推薦ループ・行動/マッチ判定・並列化）は
contrast-effect-jsai2026 と同一。以下の 3 手法を同一データ・同一シードで比較する:

  - greedy   : 貪欲方策（即時マッチング確率 P_user * P_comp の最大化）
  - surrogate: 先行研究の代理目的関数 [西村+ 2025]（γ はコマンドラインで指定）
  - proposed : 本研究の状態適応的方策（価値反復による最適価値関数を使用、
               調整パラメータなし）

定式化（論文 v3）: 離脱のみを吸収状態とし、マッチはユーザを除外しない。
応募から成立までのリードタイムにより行動時点で成立可否を観測できないため、
行動時点で P_comp（最終成立の期待値）ぶんのマッチを計上する。
環境は decay_flag（動的/静的）と churn_strength（離反の強さ）で指定する。
"""

import argparse
import logging
import os
import time
import random
import json
import numpy as np
from datetime import datetime
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 親ディレクトリ（src）をパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.utils import (
    calculate_user_state_score,
    get_user_state_score_delta,
    churn_probability,
    selection_factor,
)

from config import (
    USER_NUM,
    ITEM_NUM,
    TOP_K,
    EXPERIMENT_STEPS,
    TRIAL_NUM,
    RANDOM_SEED,
    RESULTS_DIR,
    LOGS_DIR,
    MAX_ACTION_COUNT,
    MAX_STEP_COUNT,
    ModelConfig,
)

from models.models import User, Item
from planning.value_iteration import get_or_compute_value_tables

# プロセスごとの価値関数テーブルキャッシュ
_VTABLE_CACHE = {}


def load_value_table(path: str) -> np.ndarray:
    """価値関数テーブルをロード（プロセス内でメモ化）"""
    if path not in _VTABLE_CACHE:
        _VTABLE_CACHE[path] = np.load(path)["V"]
    return _VTABLE_CACHE[path]


def load_data_fast(trial: int) -> tuple:
    """
    実験用ユーザ・求人データを高速ロード（JSAI 2026 版と同一）
    """
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    user_file = os.path.join(project_root, "data", f"user_{trial}.csv")
    user_df = pd.read_csv(user_file)

    users = []
    for i, row in user_df.iterrows():
        if i >= USER_NUM:
            break
        user = User(
            id=i,
            action_count=int(row["action_counts"]),
            last_action_step=int(row["steps_since_last_action"]),
            finished=False,
        )
        users.append(user)

    item_file = os.path.join(project_root, "data", f"item_{trial}.csv")
    item_df = pd.read_csv(item_file)

    items_dict = {}
    for _, row in item_df.iterrows():
        user_id = int(row["user"])
        step = int(row["step"])
        item_id = int(row["item"])

        if item_id < ITEM_NUM:
            key = (user_id, step)
            if key not in items_dict:
                items_dict[key] = []
            item = Item(
                user=user_id,
                step=step,
                item=item_id,
                p_user=float(row["p_user"]),
                p_comp=float(row["p_comp"]),
            )
            items_dict[key].append(item)

    return users, items_dict


def select_items(method, user, user_items, step, gamma_val, decay_flag,
                 churn_strength, selection_strength, v_table):
    """
    手法に応じて提示求人（TOP_K 件）を選択

    greedy   : argmax P_user * P_comp
    surrogate: argmax P_user * P_comp + γ * P_user * Δ(s_u)   [西村+ 2025]
    proposed : argmax { a_j P_comp(j) + (1-c)[a_j V(F+1, 0)
                         + (1 - a_j) V(F, R+1)] }              [本研究]
               = argmax P_user(j) (P_comp(j) + (1-c) ΔV(s))
               （TOP_K = 1 ではベルマン方程式右辺の厳密な最大化と一致）
    """
    if not user_items:
        return []

    if method == "greedy":
        key_fn = lambda x: x.p_user * x.p_comp

    elif method == "surrogate":
        delta = get_user_state_score_delta(user.action_count, user.last_action_step)
        key_fn = lambda x: (
            x.p_user * x.p_comp + gamma_val * x.p_user * delta
        )

    elif method == "proposed":
        f = min(user.action_count, MAX_ACTION_COUNT)
        r = min(user.last_action_step, MAX_STEP_COUNT)
        remaining = EXPERIMENT_STEPS - step - 1  # 今日の遷移後に残るステップ数
        v_next = v_table[remaining]
        va = float(v_next[min(f + 1, MAX_ACTION_COUNT), 0])
        vr = float(v_next[f, min(r + 1, MAX_STEP_COUNT)])
        c = churn_probability(f, r, churn_strength)
        if decay_flag:
            mult = calculate_user_state_score(f, r)
        else:
            mult = 1.0

        h_sel = selection_factor(f, selection_strength)

        def key_fn(x):
            a = min(x.p_user * mult, 1.0)
            # v3: 報酬は行動時点で計上（遷移と独立）
            return a * x.p_comp * h_sel + (1 - c) * (a * va + (1 - a) * vr)

    else:
        raise ValueError(f"未知の手法: {method}")

    if TOP_K == 1:
        return [max(user_items, key=key_fn)]
    return sorted(user_items, key=key_fn, reverse=True)[:TOP_K]


def run_single_trial(trial_params):
    """
    単一試行の実験を実行

    Args:
        trial_params: (trial_num, method, gamma_val, decay_flag,
                       churn_strength, v_table_path)
    Returns:
        dict: 試行結果
    """
    (trial_num, method, gamma_val, decay_flag,
     churn_strength, selection_strength, v_table_path) = trial_params

    # 試行ごとに固定乱数シードを設定（JSAI 2026 版と同一）
    trial_seed = RANDOM_SEED + trial_num
    random.seed(trial_seed)
    np.random.seed(trial_seed)

    users, items_dict = load_data_fast(trial_num)
    v_table = load_value_table(v_table_path) if method == "proposed" else None

    trial_results = {"user_action": 0, "match": 0, "churned": 0}

    for step in range(EXPERIMENT_STEPS):
        step_score = {"user_action": 0, "match": 0}

        for user in users:
            if user.finished or user.churned:
                continue

            # 開始時点の状態（離反判定は計画側と同じくこの状態で行う）
            f0, r0 = user.action_count, user.last_action_step

            user_items = items_dict.get((user.id, step), [])
            top_k_items = select_items(
                method, user, user_items, step, gamma_val,
                decay_flag, churn_strength, selection_strength, v_table,
            )

            for item in top_k_items:
                user_status = calculate_user_state_score(
                    user.action_count, user.last_action_step
                )

                # decay_flag に基づいて確率計算方法を切り替え（JSAI 2026 版と同一）
                if decay_flag:
                    p_user_adjusted = item.p_user * (
                        user_status ** ModelConfig.SCORE_ADJUSTMENT["p_user"]
                    )
                    p_comp_adjusted = item.p_comp
                else:
                    p_user_adjusted = item.p_user
                    p_comp_adjusted = item.p_comp
                # 選抜効果（本研究で追加、デフォルト無効）
                if selection_strength > 0:
                    p_comp_adjusted *= selection_factor(
                        f0, selection_strength)

                # ユーザアクション判定
                if random.random() < p_user_adjusted:
                    step_score["user_action"] += 1
                    user.action_count += 1
                    user.last_action_step = 0

                    # マッチング判定（v3: 行動時点で最終成立を判定・計上し、
                    # ユーザは離脱まで活動を継続する）
                    if random.random() < p_comp_adjusted:
                        step_score["match"] += 1
                else:
                    user.last_action_step += 1

            # 離反判定（v3: 唯一の吸収状態）
            if churn_strength > 0:
                if random.random() < churn_probability(f0, r0, churn_strength):
                    user.churned = True
                    trial_results["churned"] += 1

        for phase in ["user_action", "match"]:
            trial_results[phase] += step_score[phase]

    return trial_results


def run_experiment_parallel(method, gamma_val, decay_flag, churn_strength,
                            selection_strength, v_table_path, trial_num):
    """並列処理で実験を実行"""
    print(f"並列実験を開始: method={method}, γ={gamma_val}, "
          f"decay={decay_flag}, churn={churn_strength}")

    start_time = time.time()
    all_results = {
        "method": method,
        "gamma": gamma_val,
        "decay": decay_flag,
        "churn_strength": churn_strength,
        "trials": [],
    }

    num_cores = min(multiprocessing.cpu_count(), trial_num)
    print(f"使用コア数: {num_cores}")

    trial_params = [
        (trial, method, gamma_val, decay_flag, churn_strength,
         selection_strength, v_table_path)
        for trial in range(trial_num)
    ]

    # 試行番号順に結果を保持（試行ごとの対応比較のため順序を保証）
    results_by_trial = [None] * trial_num
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {
            executor.submit(run_single_trial, p): p[0] for p in trial_params
        }
        for i, future in enumerate(as_completed(futures)):
            if i % 10 == 0:
                print(f"完了: {i}/{trial_num} 試行")
            results_by_trial[futures[future]] = future.result()
    all_results["trials"] = results_by_trial

    _finalize_results(all_results, start_time)
    return all_results


def run_experiment_sequential(method, gamma_val, decay_flag, churn_strength,
                              selection_strength, v_table_path, trial_num):
    """逐次処理で実験を実行"""
    print(f"逐次実験を開始: method={method}, γ={gamma_val}, "
          f"decay={decay_flag}, churn={churn_strength}")

    start_time = time.time()
    all_results = {
        "method": method,
        "gamma": gamma_val,
        "decay": decay_flag,
        "churn_strength": churn_strength,
        "trials": [],
    }

    for trial in range(trial_num):
        if trial % 10 == 0:
            print(f"試行 {trial + 1}/{trial_num} を開始")
        all_results["trials"].append(run_single_trial(
            (trial, method, gamma_val, decay_flag, churn_strength,
             selection_strength, v_table_path)
        ))

    _finalize_results(all_results, start_time)
    return all_results


def _finalize_results(all_results, start_time):
    """平均値と実行時間を集計"""
    average_results = {
        phase: float(np.mean([t[phase] for t in all_results["trials"]]))
        for phase in ["user_action", "match", "churned"]
    }
    all_results["average"] = average_results
    all_results["execution_time"] = time.time() - start_time
    print(f"実験完了: 平均結果 = {average_results}")
    print(f"実行時間: {all_results['execution_time']:.2f} 秒")


def run_experiment(method, gamma_val, decay_flag, churn_strength,
                   selection_strength, v_table_path, trial_num):
    """実験を実行（並列処理優先、失敗時は逐次処理にフォールバック）"""
    try:
        return run_experiment_parallel(
            method, gamma_val, decay_flag, churn_strength,
            selection_strength, v_table_path, trial_num)
    except Exception as e:
        print(f"並列処理に失敗: {e}")
        print("逐次処理にフォールバック...")
        return run_experiment_sequential(
            method, gamma_val, decay_flag, churn_strength,
            selection_strength, v_table_path, trial_num)


def env_tag(decay_flag: bool, churn_strength: float,
            selection_strength: float = 0.0) -> str:
    """環境を表すディレクトリ名"""
    sel = f"_sel{selection_strength}" if selection_strength > 0 else ""
    if not decay_flag:
        return f"static{sel}"
    return f"dynamic_churn{churn_strength}{sel}"


def method_tag(method: str, gamma_val: float) -> str:
    """手法を表すファイル名プレフィックス"""
    if method == "surrogate":
        return f"surrogate_gamma{gamma_val}"
    return method


def save_results(results, output_dir):
    """実験結果を保存"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = method_tag(results["method"], results["gamma"])
    results_file = os.path.join(output_dir, f"{tag}_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"結果を保存: {results_file}")


def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="実験実行スクリプト（OR2026a）")
    parser.add_argument(
        "--method",
        choices=["greedy", "surrogate", "proposed"],
        default="proposed",
        help="推薦手法（greedy: 貪欲, surrogate: 先行研究, proposed: 提案）",
    )
    parser.add_argument(
        "--gamma_value",
        type=float,
        default=0.1,
        help="先行研究の将来効果重み γ（method=surrogate のときのみ使用）",
    )
    parser.add_argument(
        "--churn_strength",
        type=float,
        default=0.0,
        help="離反の強さ（0 で離反なし）",
    )
    parser.add_argument(
        "--selection_strength",
        type=float,
        default=0.0,
        help="選抜効果（高望み）の強さ: P_comp / (1 + s*F)。0 で無効",
    )
    parser.add_argument(
        "--no_decay_flag",
        action="store_false",
        dest="decay_flag",
        help="Static環境（減衰なし）を指定",
    )
    parser.add_argument(
        "--trial_limit",
        type=int,
        default=None,
        help="試行回数を上書き（デフォルトはconfig.pyに従う）",
    )
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_arguments()

    trial_num = TRIAL_NUM
    if args.trial_limit:
        trial_num = min(args.trial_limit, TRIAL_NUM)
        print(f"試行回数を上書き: {trial_num}")

    print(f"実験パラメータ: method={args.method}, γ={args.gamma_value}, "
          f"decay={args.decay_flag}, churn={args.churn_strength}, "
          f"selection={args.selection_strength}")

    # 提案手法は価値関数テーブルを事前計算（データに依存しないため 1 回だけ）
    v_table_path = ""
    if args.method == "proposed":
        v_table_path = get_or_compute_value_tables(
            args.decay_flag, args.churn_strength, args.selection_strength)

    try:
        results = run_experiment(
            args.method, args.gamma_value, args.decay_flag,
            args.churn_strength, args.selection_strength,
            v_table_path, trial_num)
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        output_dir = os.path.join(
            project_root, "results",
            env_tag(args.decay_flag, args.churn_strength,
                    args.selection_strength))
        save_results(results, output_dir)
        print("実験が正常に完了しました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise

    return 0


if __name__ == "__main__":
    exit(main())
