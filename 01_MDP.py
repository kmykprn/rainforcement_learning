"""
マルコフ決定過程の実装
"""

import random
from common.env import ENV_RANKS
from common.transit import transit_even as transit_func

from typing import List, Tuple

env = ENV_RANKS()

 # 初期位置を定義(要素が0から始めると、envの範囲外を指定する場合があるので変更しない)
start_pos: Tuple[int, int] = (1, 1)

# 現在の状態を保持
now_state_pos: Tuple[int, int] = start_pos

# 行動リストを定義
actions: List[str] = ['up', 'down', 'left', 'right']

# ゴールまでに通過したパスを保存するリスト
path_through: List[Tuple[int, int]] = [now_state_pos]

# マルコフ決定過程のメインループ
while True:

    # 遷移確率を計算
    transit_probs: List[float] = transit_func(actions)

    # 遷移確率に基づき行動を1つ選択
    action: str = random.choices(actions, k=1, weights=transit_probs)[0]

    # 行動に基づき、遷移場所の座標を取得
    new_state_pos: Tuple[int, int] = env.get_new_state_pos(now_state_pos=now_state_pos, action=action)

    # 即時報酬を獲得
    r: int = env.reward_func(new_state_pos=new_state_pos)

    # ゴール（即時報酬が1の地点）の場合、ループを抜ける
    if r == 1:
        now_state_pos = new_state_pos
        path_through.append(now_state_pos)
        break
    # 壁（即時報酬が-2の地点）の場合、座標は更新しない
    elif r == -2:
        continue
    # ゴールでも壁でもない場合は座標を更新
    else:
        now_state_pos = new_state_pos
        path_through.append(new_state_pos)

# ゴールまでに通過したパスを表示
print(path_through)