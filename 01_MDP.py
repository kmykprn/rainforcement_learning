"""
マルコフ決定過程の実装
"""

import random
from common.env import EnvRanks
from common.transit import Transit

from typing import List, Tuple


def get_new_state(current_state: Tuple[int, int], actions: List[str]) -> Tuple[int, int]:
    """
    状態と行動を受け取り、次の状態を返す関数

    Args:
        current_state:
            現在の状態（座標）
        actions:
            行動のリスト

    Returns:
        次の状態（座標）
    """

    # 遷移確率を計算
    transit_probs: List[float] = Transit.transit_even(actions)

    # 遷移確率に基づき、行動を選択
    action: str = random.choices(actions, k=1, weights=transit_probs)[0]

    # 行動に基づき、次の状態を取得
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


def main(env: EnvRanks, actions: List[str]):

    # 初期位置を定義(要素が0から始めると、envの範囲外を指定する場合があるので変更しない)
    current_state: Tuple[int, int] = (1, 1)

    # ゴールまでに通過したパスを保存するリスト
    path_through: List[Tuple[int, int]] = [current_state]

    # マルコフ決定過程のメインループ
    while True:

        # 遷移関数に基づき、次の状態を取得
        new_state: Tuple[int, int] = get_new_state(current_state=current_state, actions=actions, env=env)

        # 即時報酬を獲得
        r: int = env.reward_func(new_state)

        # ゴール（即時報酬が1の地点）の場合、ループを抜ける
        if r == 1:
            current_state = new_state
            path_through.append(current_state)
            break
        # 壁（即時報酬が-2の地点）の場合、座標は更新しない
        elif r == -2:
            continue
        # ゴールでも壁でもない場合は座標を更新
        else:
            current_state = new_state
            path_through.append(new_state)

    # ゴールまでに通過したパスを表示
    print(path_through)


if __name__ == '__main__':

    ENV = EnvRanks() # 環境を定義
    ACTIONS: List[str] = ['up', 'down', 'left', 'right'] # 行動リストを定義
    main(env=ENV, actions=ACTIONS) # マルコフ決定過程の実施