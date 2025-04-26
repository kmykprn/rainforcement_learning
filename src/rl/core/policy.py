import random
from typing import List, Dict, Tuple
from rl.utils.math_utils import softmax


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

    def to_onehot(self, probs: List[float]) -> List[float]:
        """
        最も行動価値が高い要素のベクトルを1に, それ以外を0に変換するベクトル

        Args:
            probs: 行動価値のリスト
        Returns:
            probs_onehot: 最も行動価値が高い要素を1.0に、それ以外を0.0に変換したベクトル
        """
        max_prob = max(probs)
        probs_onehot = [1.0 if p == max_prob else 0.0 for p in probs]
        return probs_onehot

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

    def random_probs_nn(self, num_actions: int) -> List[float]:
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

    def epsilon_greedy_nn(
        self,
        action_probs: List[float],
        epsilon: float = 0.8,
    ) -> List[float]:
        """
        e-greedy法を実行する関数。
        epsilonの割合でランダムな行動を取り、1-epsilonの行動で最もQ値の高い行動を取る。

        Args:
            action_probs:
                行動ごとの価値（行動確率）
            epsilon:
                探索, 活用の割合

        Returns:
            action_probs:
                行動ごとの実行確率
        """
        num_actions: int = len(action_probs)
        # 探索
        if random.random() <= epsilon:
            # ランダムな確率を設定
            return self.random_probs_nn(num_actions)
        # 活用
        else:
            # 現在の状態で、最も価値の高い行動を選択
            if action_probs and sum(action_probs) != 0:
                return self.to_onehot(action_probs)
            # ランダムな確率を設定
            else:
                return self.random_probs_nn(num_actions)

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
