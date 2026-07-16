#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人工データ生成スクリプト（2段階モデル版）
シミュレーション用のユーザデータと求人データを生成
"""

import os
import numpy as np
import pandas as pd
import argparse
from typing import Tuple

# 設定定数
USER_NUM = 1000
ITEM_NUM = 10
EXPERIMENT_STEPS = 50
RANDOM_SEED = 42
TRIAL_NUM = 100


def generate_user_data(trial_num: int, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    人工ユーザデータを生成
    
    Args:
        trial_num: 試行番号（再現性のため）
        random_seed: 基準乱数シード
        
    Returns:
        ユーザ初期化データを含むDataFrame
    """
    # 試行番号に基づいてシードを設定
    np.random.seed(random_seed + trial_num)
    
    # 初期ユーザ状態を生成
    action_counts = np.random.randint(0, 51, size=USER_NUM)  # 0-50
    steps_since_last_action = np.random.randint(0, 51, size=USER_NUM)  # 0-50
    
    user_df = pd.DataFrame({
        'user_id': range(USER_NUM),
        'action_counts': action_counts,
        'steps_since_last_action': steps_since_last_action
    })
    
    return user_df


def generate_item_data(trial_num: int, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    人工求人データを生成（2段階モデル版）
    P_userとP_compに負の相関を持たせる
    
    Args:
        trial_num: 試行番号（再現性のため）
        random_seed: 基準乱数シード
        
    Returns:
        求人確率データを含むDataFrame
    """
    # 試行番号に基づいてシードを設定
    np.random.seed(random_seed + trial_num)
    
    items_list = []
    
    for user in range(USER_NUM):
        for step in range(EXPERIMENT_STEPS):
            for item in range(ITEM_NUM):
                # P_userを生成 [0, 0.3]（recsys準拠に戻す）
                p_user = np.random.uniform(0, 0.3)
                
                # P_compを生成 [0, 0.3] * [0, 0.3]（recsysのPH2*PH3を再現）
                # 2つの独立な確率変数の積とする
                p_comp = np.random.uniform(0, 0.3) * np.random.uniform(0, 0.3)
                
                items_list.append({
                    'user': user,
                    'step': step,
                    'item': item,
                    'p_user': p_user,
                    'p_comp': p_comp
                })
    
    return pd.DataFrame(items_list)


def save_data(user_df: pd.DataFrame, item_df: pd.DataFrame, 
              trial_num: int, output_dir: str = "data") -> None:
    """
    生成したデータをCSVファイルに保存
    
    Args:
        user_df: ユーザデータDataFrame
        item_df: 求人データDataFrame
        trial_num: ファイル名に使用する試行番号
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    user_file = os.path.join(output_dir, f"user_{trial_num}.csv")
    item_file = os.path.join(output_dir, f"item_{trial_num}.csv")
    
    user_df.to_csv(user_file, index=False)
    item_df.to_csv(item_file, index=False)
    
    print(f"ユーザデータを保存: {user_file}")
    print(f"求人データを保存: {item_file}")


def generate_all_trials(output_dir: str = "data") -> None:
    """
    全試行分のデータを生成
    
    Args:
        output_dir: 出力ディレクトリ
    """
    print(f"{TRIAL_NUM} 試行分のデータを生成中...")
    
    for trial in range(TRIAL_NUM):
        if trial % 10 == 0:
            print(f"試行 {trial + 1}/{TRIAL_NUM} を生成中")
            
        user_df = generate_user_data(trial)
        item_df = generate_item_data(trial)
        save_data(user_df, item_df, trial, output_dir)
    
    print("データ生成完了！")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="実験用人工データを生成")
    parser.add_argument("--output_dir", default="data", 
                       help="生成データの出力ディレクトリ")
    parser.add_argument("--trial", type=int, default=None,
                       help="特定の試行のみ生成（デフォルト: 全試行）")
    
    args = parser.parse_args()
    
    if args.trial is not None:
        print(f"試行 {args.trial} のデータを生成中")
        user_df = generate_user_data(args.trial)
        item_df = generate_item_data(args.trial)
        save_data(user_df, item_df, args.trial, args.output_dir)
    else:
        generate_all_trials(args.output_dir)


if __name__ == "__main__":
    main()
