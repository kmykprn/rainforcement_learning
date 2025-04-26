import random
from typing import List, Dict, Tuple
from rl.utils.math_utils import softmax


class Policy:
    def q_max_to_onehot(self, q_values: List[float]) -> List[float]:
        """
        最も行動価値が高い要素のベクトルを1.0に, それ以外を0.0に変換するベクトル

        Args:
            q_values: 行動ごとのQ値
        Returns:
            q_onehot: 最も行動価値が高い要素を1.0に、それ以外を0.0に変換したベクトル
        """
        max_value = max(q_values)
        q_onehot = [1.0 if v == max_value else 0.0 for v in q_values]
        return q_onehot

    def random_probs(self, num_actions: int) -> List[float]:
        """
        行動ごとに、ランダムな確率（足して1になるように正規化済み）を割り当てる関数。

        Args:
            num_actions: 取りうる行動の個数

        Returns:
            action_probs: 行動ごとの均等な実行確率
        """
        # 行動ごとに、ランダムな確率を設定
        random_list: List[float] = [random.random() for _ in range(num_actions)]

        # 足して1になるように正規化
        total: float = sum(random_list)
        action_probs: List[float] = [r / total for r in random_list]

        return action_probs

    def epsilon_greedy(
        self,
        q_values: List[float],
        epsilon: float = 0.2,
    ) -> List[float]:
        """
        e-greedy法を実行する関数。
        epsilonの割合でランダムな行動を取り、1-epsilonの行動で最もQ値の高い行動を取る。

        Args:
            q_values:
                行動ごとのQ値
            epsilon:
                探索の割合(=epsilon), 活用の割合(=1-epsilon)

        Returns:
            action_probs:
                行動ごとの実行確率
        """
        num_actions: int = len(q_values)
        # 探索
        if random.random() <= epsilon:
            return self.random_probs(num_actions)
        # 活用
        else:
            if q_values and sum(q_values) != 0:
                # 現在の状態で、最も価値の高い行動を1.0とした行動確率を取得
                return self.q_max_to_onehot(q_values)
            else:
                return self.random_probs(num_actions)

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
