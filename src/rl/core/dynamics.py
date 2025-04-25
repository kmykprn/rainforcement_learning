from typing import Tuple


class Ranks:
    def get_new_state(
        self, current_state: Tuple[int, int], action: str
    ) -> Tuple[int, int]:
        """
        行列形式において、現在の状態と行動をもとに、次の状態を決定する関数

        Args:
            current_state:
                現在の状態（座標）
            actions:
                行動のリスト

        Returns:
            次の状態（座標）
        """

        # 行動に基づき、次の状態を取得
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
