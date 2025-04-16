from typing import Tuple, List


class EnvRanks:
    """
    行列形式の環境と、環境に基づく報酬関数を定義
    """

    def __init__(self, ranks: List[List[int]]=None):
        """
        環境の行列を定義

        Args:
            ranks:
                2次元配列

        Returns:
            None
        """

        if ranks:
            self.ranks = ranks
        else:
            self.ranks = [
                [-2, -2, -2, -2, -2, -2],
                [-2,  0,  0,  0,  0, -2],
                [-2,  0, -2,  0,  0, -2],
                [-2,  0,  0, -1,  10, -2],
                [-2, -2, -2, -2, -2, -2],
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
    def get_new_state_pos(current_state: Tuple[int, int], action: str) -> Tuple[int, int]:
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

        if action == 'up':
            row -= 1
        if action == 'down':
            row += 1
        if action == 'left':
            col -= 1
        if action == 'right':
            col += 1

        new_state: Tuple[int, int] = (row, col)

        return new_state