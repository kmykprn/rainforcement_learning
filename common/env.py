from typing import Tuple


class ENV_RANKS:
    """
    行列形式の環境と、環境に基づく報酬関数を定義
    """

    def __init__(self, env=None):

        if env:
            self.env = env
        else:
            self.env = [
                [-2, -2, -2, -2, -2, -2],
                [-2,  0,  0,  0,  0, -2],
                [-2,  0, -2,  0,  0, -2],
                [-2,  0,  0, -1,  1, -2],
                [-2, -2, -2, -2, -2, -2],
            ]

    def reward_func(self, new_state_pos: Tuple[int, int]) -> int:
        """
        行列環境における即時報酬関数

        Args:
            new_state_pos:
                次の状態（座標）

        Returns:
            次の状態における環境からの報酬
        """
        row = new_state_pos[0]
        col = new_state_pos[1]
        return self.env[row][col]

    @staticmethod
    def get_new_state_pos(now_state_pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        行動に基づき、状態(場所)を更新

        Args:
            now_state_pos:
                現在の状態（座標）
            action:
                行動

        Returns:
            new_state_pos:
                次の状態（座標）
        """
        row = now_state_pos[0]
        col = now_state_pos[1]

        if action == 'up':
            row -= 1
        if action == 'down':
            row += 1
        if action == 'left':
            col -= 1
        if action == 'right':
            col += 1

        new_state_pos = (row, col)
        return new_state_pos