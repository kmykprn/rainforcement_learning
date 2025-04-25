import random
from typing import List, Dict, Tuple
from utils.math_utils import softmax


class Policy:
    def random_probs(self, actions: List[str]) -> List[float]:
        """
        行動ごとに、ランダムな確率（足して1になるように正規化済み）を割り当てる関数。

        Args:
            actions:
                行動のリスト

        Returns:
            action_probs:
                行動ごとの実行確率
        """
        # 行動ごとに、ランダムな確率を設定
        random_list: List[float] = [random.random() for _ in actions]

        # 足して1になるように正規化
        total: float = sum(random_list)
        action_probs: List[float] = [r / total for r in random_list]

        return action_probs

    def max_value_probs(
        self,
        current_state: Tuple[int, int],
        actions: List[str],
        Q: Dict[Tuple[int, int], Dict[str, float]],
    ) -> List[float]:
        """
        価値が最大の行動を1.0, それ以外を0.0とする行動確率を返す関数

        Args:
            Q:
                Qテーブル
            current_state:
                現在の状態
            actions:
                行動候補

        Returns:
            action_probs:
                行動ごとの実行確率
        """

        # 行動と行動確率を取り出し
        q_current_state: Dict[str, float] = Q[current_state]
        action_list: List[str] = list(q_current_state.keys())
        action_probs_list: List[float] = list(q_current_state.values())

        # 行動確率の中から、最も行動確率が高い要素のインデックスを取得
        max_q_action_probs: float = max(action_probs_list)
        max_index: int = action_probs_list.index(max_q_action_probs)

        # 対応する行動（キー）を取得
        max_q_action: str = action_list[max_index]

        # 行動価値が高いインデックスを1.0に、それ以外を0.0にしたリストを生成
        action_probs = [0.0] * len(actions)
        idx = actions.index(max_q_action)
        action_probs[idx] = 1.0

        return action_probs

    def epsilon_greedy(
        self,
        current_state: Tuple[int, int],
        actions: List[str],
        Q: Dict[Tuple[int, int], Dict[str, float]],
        epsilon: float = 0.8,
    ) -> List[float]:
        """
        各行動ごとに、行動確率を決定する関数

        Args:
            state:
                現在の状態
            actions:
                行動のリスト
            Q:
                ある状態(座標)において、行動を取ったときの評価値。
                例.
                    Q = {
                        (1, 1): {'up': 0.2, 'down': 0.3, 'left': 0.1, 'right': 0.5},
                        ...
                    }
            epsilon:
                探索, 活用の割合

        Returns:
            action_probs:
                行動ごとの実行確率
        """
        # 探索
        if random.random() <= epsilon:
            # ランダムな確率を設定
            return self.random_probs(actions)
        # 活用
        else:
            if (current_state in Q) and sum(Q[current_state].values()) != 0:
                # 現在の状態で、最も価値の高い行動を選択
                return self.max_value_probs(current_state, actions, Q)
            else:
                # ランダムな確率を設定
                return self.random_probs(actions)

    def q_probs(
        self,
        current_state: Tuple[int, int],
        Q: Dict[Tuple[int, int], Dict[str, float]],
    ) -> List[float]:
        """
        Q値を行動確率に変換する関数。

        Args:
            current_state:
                現在の状態

            Q:
                Qテーブル

        Returns:
            action_probs:
                行動ごとの実行確率
        """

        # softmaxを使用し、価値を行動確率に変換
        action_probs: List[float] = softmax(list(Q[current_state].values()))

        return action_probs
