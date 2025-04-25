import yaml
from typing import Tuple, List, Optional


class EnvRanks:
    """
    行列形式の環境と、環境に基づく報酬関数を定義
    """

    def __init__(self, ranks: Optional[List[List[int]]] = None):
        """
        環境の行列を定義

        Args:
            ranks:
                2次元配列

        Returns:
            None
        """

        # 設定ファイルを読み込み
        with open("config/base.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        WALL = config["env"]["wall_reward"]
        GOAL = config["env"]["goal_reward"]
        PATH = config["env"]["path_reward"]
        TRAP = config["env"]["trap_reward"]

        # 環境を定義
        if ranks:
            self.ranks = ranks
        else:
            self.ranks = [
                [WALL, WALL, WALL, WALL, WALL, WALL],
                [WALL, PATH, PATH, PATH, TRAP, WALL],
                [WALL, PATH, WALL, PATH, PATH, WALL],
                [WALL, PATH, PATH, TRAP, GOAL, WALL],
                [WALL, WALL, WALL, WALL, WALL, WALL],
            ]

    def reward_func(self, new_state: Tuple[int, int]) -> int:
        """
        行列環境における即時報酬関数

        Args:
            new_state_pos:
                次の状態（座標）

        Returns:
            次の状態における環境からの報酬
        """
        row = new_state[0]
        col = new_state[1]
        return self.ranks[row][col]

    def get_states(self) -> List[Tuple[int, int]]:
        """
        行列環境のすべての状態（座標）を取得する関数。

        Returns:
            states:
                例. [(0, 0), (0, 1), ..., (4, 5)]
        """
        rows = len(self.ranks)
        cols = len(self.ranks[0])

        states: List[Tuple[int, int]] = []
        for col in range(cols):
            for row in range(rows):
                states.append((row, col))

        return states
