#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ユーザ状態スコアの3D可視化スクリプト
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_fontja
from mpl_toolkits.mplot3d import Axes3D

# フォントサイズ設定（日本語フォントはmatplotlib_fontjaが設定）
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
})

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.utils import calculate_user_state_score


def create_user_state_3d_plot(output_path: str):
    """
    ユーザ状態スコアの3D可視化を作成
    
    Args:
        output_path: 出力ファイルパス
    """
    # グリッドを生成
    frequency = np.arange(0, 101, 5)  # 0-100 (累積行動数)
    recency = np.arange(0, 101, 5)    # 0-100 (経過ステップ数)
    F, R = np.meshgrid(frequency, recency)
    
    # Static環境とDynamic環境のスコアを計算
    scores_static = np.ones_like(F, dtype=float)
    scores_dynamic = np.zeros_like(F, dtype=float)
    
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            scores_dynamic[i, j] = calculate_user_state_score(F[i, j], R[i, j])
    
    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                    subplot_kw={'projection': '3d'})
    
    # Static環境
    ax1.plot_surface(F, R, scores_static, cmap='coolwarm', alpha=0.8)
    ax1.set_xlabel('Frequency (累積行動数)')
    ax1.set_ylabel('Recency (経過ステップ)')
    ax1.set_zlabel('将来マッチング確率')
    ax1.set_title('(a) 静的環境')
    ax1.set_zlim(0, 2.2)
    ax1.view_init(elev=30, azim=120)  # 視点を180度回転
    
    # Dynamic環境
    ax2.plot_surface(F, R, scores_dynamic, cmap='coolwarm', alpha=0.8)
    ax2.set_xlabel('Frequency (累積行動数)')
    ax2.set_ylabel('Recency (経過ステップ)')
    ax2.set_zlabel('将来マッチング確率')
    ax2.set_title('(b) 動的環境')
    ax2.set_zlim(0, 2.2)
    ax2.view_init(elev=30, azim=120)  # 視点を180度回転
    
    plt.subplots_adjust(left=0.02, right=0.98, wspace=0)
    plt.savefig(output_path, dpi=300, pad_inches=0.5)
    plt.close()
    
    print(f"ユーザ状態スコアの3D可視化を保存: {output_path}")


def main():
    """メイン関数"""
    print("=== ユーザ状態スコアの可視化を生成 ===")
    
    # プロジェクトルートを取得
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3D可視化を生成
    output_path = os.path.join(output_dir, "user_state_score_3d.pdf")
    create_user_state_3d_plot(output_path)
    
    print("可視化の生成が完了しました")


if __name__ == "__main__":
    main()
