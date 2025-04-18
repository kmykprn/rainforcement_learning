import yaml
from typing import Tuple, List


class EnvRanks:
    """
    行列形式の環境と、環境に基づく報酬関数を定義
    """

    def __init__(self, ranks: List[List[int]] = None):
        """
        環境の行列を定義

        Args:
            ranks:
                2次元配列

        Returns:
            None
        """

        # 設定ファイルを読み込み
        with open("02_MonteCarlo/config.yaml", "r", encoding="utf-8") as file:
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

    @staticmethod
    def get_new_state_pos(
        current_state: Tuple[int, int], action: str
    ) -> Tuple[int, int]:
        """
        行動に基づき、次の状態を取得

        Args:
            current_state:
                現在の状態（座標）
            action:
                行動

        Returns:
            new_state:
                新たな状態（座標）
        """
        row = current_state[0]
        col = current_state[1]

        if action == "up":
            row -= 1
        if action == "down":
            row += 1
        if action == "left":
            col -= 1
        if action == "right":
            col += 1

        new_state: Tuple[int, int] = (row, col)

        return new_state
