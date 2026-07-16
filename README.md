# 離反を考慮した状態適応的方策による双方向推薦 実験コード

本リポジトリは、論文「離反を考慮した状態適応的方策による双方向推薦」
（日本オペレーションズ・リサーチ学会 2026年秋季研究発表会）の実験コードです。

先行研究のリポジトリ
[contrast-effect-jsai2026](https://github.com/nnnnishi/contrast-effect-jsai2026)
の構成・実験設定を踏襲しており、**データ生成・ユーザ状態スコア・確率モデル・
シミュレータの骨格・乱数シードはすべて JSAI 2026 版と同一**です。
差分は本研究の貢献部分（離反モデル・価値反復による状態適応的方策・手法比較）のみです。

## 概要

サービス利用期間中のマッチング率最大化問題を、マッチ成立を吸収状態とする
マルコフ決定過程（MDP）として定式化します。マッチ成立時に報酬 1 を一度だけ
与えることで、価値関数は計画期間内にマッチへ至る確率そのものとなり、
**目的関数が評価指標（マッチ率）と厳密に一致**します。
累積行動数 F と最終行動からの経過ステップ数 R からなる 2 次元状態に対する
価値反復法（後ろ向き帰納法）で最適方策を求めます。

## 主な特徴

- **MDP 定式化**: マッチ成立を吸収状態とし、目的関数 = マッチ率
- **価値反復法**: 低次元の表形式状態 (F, R) に対する決定論的な解導出
- **調整パラメータなし**: 先行研究の将来効果重み γ（λ）が不要
- **離反モデル**: ユーザ状態に応じた離反確率 P_churn（本研究で追加）
- **フェアな比較**: 3手法（貪欲 / 先行研究 / 提案）を同一データ・同一シードで比較

## 先行研究（JSAI 2026 版）との対応

| 項目 | JSAI 2026 版 | 本リポジトリ |
| --- | --- | --- |
| データ生成 | `create_data.py` | **同一**（コード・シードとも） |
| ユーザ状態スコア s(F, R) | `utils.py` | **同一** |
| 動的環境（P_user × s） | `--no_decay_flag` で切替 | **同一** |
| 離反 | なし | **追加**: P_churn = strength × (1 − s/2) |
| 貪欲方策 | `--gamma_value 0.0` | `--method greedy` |
| 代理目的関数 [西村+ 2025] | `--gamma_value γ` | `--method surrogate --gamma_value γ` |
| 提案手法（価値反復） | ― | `--method proposed` |

（論文・スライドの λ はコード上の γ に対応します）

## ディレクトリ構成

```
state-adaptive-policy-or2026a/
├── src/
│   ├── config.py                  # 実験パラメータ設定（基本値は JSAI 版と同一）
│   ├── models/
│   │   └── models.py              # ユーザ・求人データクラス（離反フラグを追加）
│   ├── utils/
│   │   └── utils.py               # 状態スコア（JSAI 版と同一）+ 離反確率
│   ├── data_processing/
│   │   └── create_data.py         # 人工データ生成（JSAI 版と同一）
│   ├── planning/
│   │   ├── value_iteration.py     # 価値反復（本研究のコア）
│   │   └── smoothing.py           # 単調性制約による平滑化（論文 3.2 節）
│   ├── experiments/
│   │   └── experiment.py          # メイン実験スクリプト（3手法比較）
│   └── visualization/
│       ├── visualize_user_state.py     # ユーザ状態の3D可視化（JSAI 版と同一）
│       ├── visualize_value_function.py # V と ΔV の可視化
│       ├── plot_ratio_vs_churn.py      # 離反強度×対貪欲比（発表メイン図・左）
│       └── plot_gamma_sweep.py         # γ感度プロット（発表メイン図・右）
├── data/                          # 生成された実験データ
├── results/                       # 実験結果・価値関数テーブル・プロット
├── requirements.txt
├── run_full_experiment.sh         # 全実験実行スクリプト
└── README.md
```

## インストール

```bash
git clone [REPOSITORY_URL]
cd state-adaptive-policy-or2026a
pip install -r requirements.txt
```

## クイックスタート

### 全実験の一括実行

```bash
./run_full_experiment.sh
```

### 個別実行

```bash
# 1. 人工データの生成（JSAI 2026 版と同一）
cd src/data_processing && python create_data.py --output_dir ../../data && cd ../..

# 2. 価値関数テーブルの事前計算（環境ごとに1回、データに依存しない）
python src/planning/value_iteration.py --churn_strength 0.05

# 3. 実験の実行（同一環境で3手法を比較）
python src/experiments/experiment.py --method greedy    --churn_strength 0.05
python src/experiments/experiment.py --method proposed  --churn_strength 0.05
python src/experiments/experiment.py --method surrogate --gamma_value 0.1 --churn_strength 0.05

# 静的環境（サニティチェック: 提案手法は貪欲と厳密に一致）
python src/experiments/experiment.py --method greedy   --no_decay_flag
python src/experiments/experiment.py --method proposed --no_decay_flag

# 4. 可視化
python src/visualization/plot_ratio_vs_churn.py
python src/visualization/plot_gamma_sweep.py --churn_strength 0.05
python src/visualization/visualize_value_function.py --churn_strength 0.05
```

## 実験パラメータ

主要パラメータは `src/config.py` で設定可能（JSAI 2026 版と同一の値）:

| パラメータ | デフォルト値 | 説明 |
| --- | --- | --- |
| USER_NUM | 1000 | シミュレーション対象のユーザ数 |
| ITEM_NUM | 10 | 推薦スレート内の求人数 |
| EXPERIMENT_STEPS | 50 | シミュレーションステップ数（計画期間 Dmax） |
| TRIAL_NUM | 100 | 実験試行回数 |
| RANDOM_SEED | 42 | 再現性のための乱数シード |
| CHURN_GRID | [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5] | 離反の強さの掃引グリッド |
| GAMMA_GRID | [0.01, 0.1, 1.0, 10.0] | 先行研究 γ の掃引グリッド |

## 数値実験の設定

### データ生成（JSAI 2026 版と同一）

- **求職者数**: 1,000人、**候補求人数**: 各期10件、**Top-k**: k = 1
- **行動確率**: P_user ~ U(0, 0.3)、P_comp ~ U(0, 0.3) × U(0, 0.3)
- **状態変数の初期値**: F ~ U(0, 50)、R ~ U(0, 50)
- **ユーザ状態スコア**: s = (1 + log(1+F)/log(101)) × (0.05^(1/100))^R ∈ [0, 2]

### 環境

| 環境 | 定義 | 指定方法 |
| --- | --- | --- |
| **静的環境** | 状態が行動確率に影響しない | `--no_decay_flag` |
| **動的環境** | 各期の P_user に s を乗算（JSAI 版と同一） | デフォルト |
| **動的＋離反環境** | さらに毎期 P_churn = strength × (1 − s/2) で離反 | `--churn_strength c` |
| **選抜効果（高望み）** | P_comp × 1/(1 + s_sel × F)：応募数が多いほどマッチしにくい（感度分析用） | `--selection_strength s_sel` |

### 推薦手法

- **貪欲方策**: argmax P_user × P_comp
- **先行研究**（代理目的関数 [西村+ 2025]）: argmax P_user × P_comp + γ × P_user × Δ(s)
- **提案手法**（状態適応的方策）: 価値反復で得た V を用い、ベルマン方程式右辺を最大化

  argmax { m_j + (1−c)(a_j − m_j) V(F+1, 0) + (1−c)(1 − a_j) V(F, R+1) }

  Top-k = 1 ではベルマン方程式の厳密な最大化と一致し、調整パラメータを持たない。

### フェアな比較のためのプロトコル

1. **同一データ・同一シード**: 3手法とも同じ試行データ（`data/`）と乱数シードで評価し、
   試行ごとの対貪欲比の平均と 95% 信頼区間を報告する
2. **真の確率を共有**: すべての手法が真の P_user・P_comp（と状態スコア s）へ
   アクセスでき、推定誤差の影響を受けない（方策の質のみを比較）
3. **γ はグリッド掃引**: 先行研究は γ ∈ {0.01, 0.1, 1, 10} を全環境で評価し、
   環境ごとの最良値・固定運用の両方を示せる
4. **価値関数はデータ非依存**: 提案手法の価値反復は候補求人の生成分布のみから
   事前計算され、評価データへのアクセスはない

### 選抜効果の感度分析（発表バックアップ）

元の環境では F は「活動量」としてのみ働くため、離反が強い環境では
「行動実績の多い求職者の離反期」で ΔV が最大となる。
選抜効果（`--selection_strength 0.1` など）を加えると
「F 小 × R 大（離反しかけの新規求職者）」が最優先に反転し、
方策が環境の仮定に自動適応することを確認できる:

```bash
python src/planning/value_iteration.py --churn_strength 0.3 --selection_strength 0.1
python src/visualization/visualize_value_function.py --churn_strength 0.3 --selection_strength 0.1
```

### 期待される結果

- **静的環境**: ΔV = 0 となり提案手法の方策は貪欲方策と厳密に一致
  （対貪欲比 = 1.000。共通乱数により数値誤差なく一致する）
- **動的環境**: 離反が強いほど提案手法の対貪欲比が拡大（0.2〜0.3 で頭打ち）
- **γ 感度**: 先行研究は最良の γ でも提案手法以下、γ を外すと貪欲未満

## 論文引用

```
@inproceedings{nishimura2026stateadaptive,
  title={離反を考慮した状態適応的方策による双方向推薦},
  author={西村直樹 and 小山田創哲},
  booktitle={日本オペレーションズ・リサーチ学会 2026年秋季研究発表会},
  year={2026}
}
```

## ライセンス

本プロジェクトはMITライセンスの下で公開されています。詳細はLICENSEファイルを参照してください。
