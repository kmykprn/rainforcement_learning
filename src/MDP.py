"""
マルコフ決定過程の実装
"""

import random
import yaml

from rl.core.env import EnvRanks
from rl.core.dynamics import Ranks
from rl.core.transition_prob import Transit

from typing import List, Tuple


def main(env: EnvRanks, actions: List[str]):

    # 初期位置を定義(要素が0から始めると、envの範囲外を指定する場合があるので変更しない)
    current_state: Tuple[int, int] = (1, 1)

    # ゴールまでに通過したパスを保存するリスト
    path_through: List[Tuple[int, int]] = [current_state]

    # 遷移確率を算出するオブジェクトを定義
    transit = Transit()

    # 現在の状態と行動をもとに、次の状態を計算するオブジェクトを呼び出し
    rank_dynamics = Ranks()

    # マルコフ決定過程のメインループ(MAX_STEPまでにゴールに到達しなければ打ち切り)
    for _ in range(MAX_STEP):

        # 遷移確率を取得
        transit_probs: List[float] = transit.transit_even(actions)

        # 遷移確率に基づき、行動を選択
        action: str = random.choices(actions, k=1, weights=transit_probs)[0]

        # 遷移関数に基づき、次の状態を取得
        new_state: Tuple[int, int] = rank_dynamics.get_new_state(
            current_state=current_state, action=action
        )

        # 1時刻後の即時報酬を獲得
        r_new_state: int = env.reward_func(new_state)

        # ゴールの場合、Qテーブルを更新してループを抜ける
        if r_new_state == GOAL_REWARD:
            current_state = new_state
            path_through.append(current_state)
            break
        # 壁（即時報酬が-2の地点）の場合、座標は更新しない
        elif r_new_state == WALL_REWARD:
            continue
        # ゴールでも壁でもない場合は座標を更新
        else:
            current_state = new_state
            path_through.append(new_state)

    # 評価（ゴールまでに通過したパスを表示）
    print(path_through)


if __name__ == "__main__":

    # 設定ファイルを読み込み
    with open("config/base.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 環境を定義
    ENV = EnvRanks()
    GOAL_REWARD: int = config["env"]["goal_reward"]
    WALL_REWARD: int = config["env"]["wall_reward"]

    # 行動リストを定義
    ACTIONS: List[str] = config["actions"]

    # 試行の打ち切り回数を定義
    MAX_STEP: int = config["learning"]["max_step"]

    # マルコフ決定過程の実施
    main(env=ENV, actions=ACTIONS)
