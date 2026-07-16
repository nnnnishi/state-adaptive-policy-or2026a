"""
実験パラメータ設定ファイル（OR2026a: 状態適応的方策版）

基本パラメータは contrast-effect-jsai2026 と同一に保ち、
離反（churn）と価値反復（value iteration）のパラメータのみ追加している。
"""

# 基本パラメータ（JSAI 2026 版と同一）
USER_NUM = 1000
ITEM_NUM = 10
TOP_K = 1
EXPERIMENT_STEPS = 50
RANDOM_SEED = 42
TRIAL_NUM = 100

# ユーザ状態の上限値（JSAI 2026 版と同一）
MAX_ACTION_COUNT = 100
MAX_STEP_COUNT = 100

# 結果保存ディレクトリ
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# ---- 離反（churn）パラメータ（本研究で追加）----
# P_churn(F, R) = churn_strength * (1 - s(F, R) / 2)
#   s は [0, 2] のユーザ状態スコア（JSAI 2026 版と同一の関数）
#   活動的なユーザほど離反しにくく、行動が止まったユーザほど離反しやすい
DEFAULT_CHURN_STRENGTH = 0.0
CHURN_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# ---- 選抜効果（高望み）パラメータ（本研究で追加、感度分析用）----
# P_comp_eff = P_comp / (1 + selection_strength * F)
#   応募数 F が多いほどマッチしにくい、という負のシグナルを表す
#   0 で無効（デフォルト）。バックアップ実験では 0.1 を使用
DEFAULT_SELECTION_STRENGTH = 0.0

# ---- 先行研究（代理目的関数）の γ グリッド（JSAI 2026 版と同一の掃引）----
GAMMA_GRID = [0.01, 0.1, 1.0, 10.0]

# ---- 価値反復（value iteration）パラメータ（本研究で追加）----
VI_NUM_CANDIDATE_SETS = 200   # 候補求人集合のモンテカルロサンプル数（全状態で共通乱数）
VI_CHUNK_SIZE = 20            # メモリ節約のためのチャンクサイズ
VI_CACHE_DIR = "results/value_tables"
VI_SEED = 12345               # 候補集合サンプリング用シード（実験データとは独立）


class ModelConfig:
    """モデル固有のパラメータ（JSAI 2026 版と同一）"""

    @staticmethod
    def baseline_score(item):
        """ベースライン（貪欲方策）スコア: 即時マッチング確率"""
        return item.p_user * item.p_comp

    @staticmethod
    def surrogate_score(item, delta, gamma_value):
        """先行研究（代理目的関数）のスコア [西村+ 2025]"""
        return (
            item.p_user * item.p_comp
            + gamma_value * item.p_user * delta
        )

    # スコア調整パラメータ: Dynamic環境でのユーザ状態による確率調整
    SCORE_ADJUSTMENT = {
        "p_user": 1,  # アクション確率の指数（s_u をそのまま乗算）
    }
