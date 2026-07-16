#!/bin/bash

# 状態適応的方策による推薦システム実験（OR2026a）- 全実験実行スクリプト
# 本スクリプトで論文の主要結果を再現できます
#
# データ生成・環境・評価プロトコルは contrast-effect-jsai2026 と同一。
# 3手法（貪欲 / 先行研究の代理目的関数 / 提案の状態適応的方策）を
# 同一データ・同一シードで比較します。

set -e  # エラー発生時に終了

CHURN_GRID="0.01 0.02 0.05 0.1 0.2 0.3 0.5"
GAMMA_GRID="0.01 0.1 1.0 10.0"

echo "=== 状態適応的方策による推薦システム実験（OR2026a）==="
echo "推定時間: システムにより 30〜60 分程度"
echo ""

# 必要なディレクトリを作成
echo "ディレクトリを作成中..."
mkdir -p data results logs results/value_tables

# Step 1: 人工データの生成（JSAI 2026 版と同一の生成コード・シード）
echo "Step 1/5: 人工データを生成中..."
cd src/data_processing
python create_data.py --output_dir ../../data
cd ../..

# Step 2: 価値関数テーブルの事前計算（データに依存しないため環境ごとに1回）
echo ""
echo "Step 2/5: 価値関数テーブルを事前計算中..."
python src/planning/value_iteration.py --no_decay_flag --churn_strength 0.0
python src/planning/value_iteration.py --churn_strength 0.0
for c in $CHURN_GRID; do
    python src/planning/value_iteration.py --churn_strength $c
done

# Step 3: 静的環境（サニティチェック: 提案手法は貪欲と厳密に一致するはず）
echo ""
echo "Step 3/5: 静的環境の実験を実行中..."
python src/experiments/experiment.py --method greedy   --no_decay_flag
python src/experiments/experiment.py --method proposed --no_decay_flag
for g in $GAMMA_GRID; do
    python src/experiments/experiment.py --method surrogate --gamma_value $g --no_decay_flag
done

# Step 4: 動的環境（離反の強さを掃引）
echo ""
echo "Step 4/5: 動的環境の実験を実行中..."
for c in 0.0 $CHURN_GRID; do
    echo "  離反の強さ = $c"
    python src/experiments/experiment.py --method greedy   --churn_strength $c
    python src/experiments/experiment.py --method proposed --churn_strength $c
    for g in $GAMMA_GRID; do
        python src/experiments/experiment.py --method surrogate \
            --gamma_value $g --churn_strength $c
    done
done

# Step 5: 可視化の生成
echo ""
echo "Step 5/5: 可視化を生成中..."
python src/visualization/visualize_user_state.py
python src/visualization/visualize_value_function.py --churn_strength 0.05
python src/visualization/plot_ratio_vs_churn.py
python src/visualization/plot_gamma_sweep.py --churn_strength 0.05

echo ""
echo "=== 実験が正常に完了しました！ ==="
echo ""
echo "結果は以下に保存されています:"
echo "  - results/static/                  : 静的環境の結果"
echo "  - results/dynamic_churn*/          : 動的（+離反）環境の結果"
echo "  - results/value_tables/            : 価値関数テーブル"
echo ""
echo "主要ファイル:"
echo "  - results/ratio_vs_churn.pdf       : 離反強度×対貪欲比（発表メイン図・左）"
echo "  - results/gamma_sweep_churn0.05.pdf: γ感度プロット（発表メイン図・右）"
echo "  - results/delta_v_heatmap_*.pdf    : ΔV ヒートマップ（方策の解釈図）"
echo "  - results/value_function_3d_*.pdf  : 価値関数の3D可視化"
echo ""
echo "期待される結果:"
echo "  - 静的環境: 提案手法は貪欲方策と厳密に一致（対貪欲比 = 1.000）"
echo "  - 動的環境: 離反が強いほど提案手法の対貪欲比が拡大"
echo "  - γ感度  : 先行研究は最良のγでも提案手法以下、γを外すと貪欲未満"
