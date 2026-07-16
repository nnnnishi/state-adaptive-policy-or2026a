"""
2段階モデル用データクラス定義（OR2026a: 離反フラグを追加）
"""

from dataclasses import dataclass


@dataclass
class User:
    """ユーザ（求職者）クラス

    Attributes:
        id: ユーザID
        action_count: 累積オンライン行動数（Frequency）
        last_action_step: 最終行動からの経過ステップ数（Recency）
        finished: マッチング成立フラグ
        churned: 離反フラグ（本研究で追加。離反後は推薦・マッチングの対象外）
    """
    id: int
    action_count: int = 0
    last_action_step: int = 0
    finished: bool = False
    churned: bool = False


@dataclass
class Item:
    """求人クラス（2段階モデル、JSAI 2026 版と同一）

    Attributes:
        user: 対象ユーザID
        step: シミュレーションステップ
        item: 求人ID
        p_user: 求職者のアクション確率（応募・閲覧など）
        p_comp: 企業の受容確率（面接承諾・内定など）
    """
    user: int
    step: int
    item: int
    p_user: float
    p_comp: float
