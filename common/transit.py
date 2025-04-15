import random
from typing import List

class Transit:
    def __init__(self):
        pass

    def transit_random(self, actions: List[str]) -> List[float]:
        """
        次の状態にランダムな確率で遷移する遷移関数

        Args:
            actions:
                取りうる行動の候補

        Returns:
            transit_probs:
                次の状態への遷移確率
        """
        random_list: List[float] = [random.random() for _ in actions]

        # 合計が1になるように正規化
        total: float = sum(random_list)
        transit_probs: List[float] = [r / total for r in random_list]
        return transit_probs


    def transit_even(actions: List[str]) -> List[float]:
        """
        次の状態に等確率で遷移する遷移確率リストを生成する関数

        Args:
            actions:
                取りうる行動の候補

        Returns:
            transit_probs:
                次の状態への遷移確率
        """
        # 1を候補数で割って等確率を計算
        prob: float = 1 / len(actions)

        # 各候補に同じ確率を割り当てたリストを返す
        transit_probs: List[float] = [prob for _ in actions]
        return transit_probs