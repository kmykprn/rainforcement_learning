from core.env import EnvRanks
from core.dynamics import Ranks

from typing import Dict, List, Tuple


def get_max_q_action(current_state: Tuple, Q: Dict[Tuple, Dict[str, float]]) -> str:
    """
    現在の状態における、価値最大の行動を取得する

    Args:
        current_state:
            現在の状態

        Q:
            Qテーブル

    Returns:
        action:
            行動
    """
    # 価値最大の行動を選択
    max_value = max(Q[current_state].values())
    action: str = [
        action for action, value in Q[current_state].items() if value == max_value
    ][0]

    return action


def evaluate_Q(
    count: int,
    env: EnvRanks,
    Q: Dict[Tuple[int, int], Dict[str, float]],
    goal_reward: int,
    wall_reward: int,
):
    """
    学習済みQテーブルで、成功/失敗を判定する関数

    Args:
        count:
            Q値の学習回数
        env:
            環境
        Q:
            Qテーブル
    """

    # スタート地点を定義
    state: Tuple[int, int] = (1, 1)

    # 通過した座標を記録
    path_through: List[Tuple[int, int]] = [state]

    # 状態と行動をもとに、次の状態を計算する関数
    rank_dynamics = Ranks()

    # 価値最大の行動を行なっても、ゴールに到達しない場合があるので、50試行で打ち止め
    for _ in range(50):

        # 価値最大の行動を選択
        action: str = get_max_q_action(state, Q)
        next_state: Tuple[int, int] = rank_dynamics.get_new_state(state, action)
        r: int = env.reward_func(next_state)

        # ゴールに到達した場合
        if r == goal_reward:
            path_through.append(next_state)
            print(f"学習回数: {count}回, 成功!")
            print(f"経路：{path_through}")
            return
        # 壁で停止した場合
        elif r == wall_reward:
            print(f"学習回数: {count}回, 失敗...")
            return
        # 進む
        else:
            path_through.append(next_state)
            state = next_state

    # 上記のいずれでも終了しなかった場合（未到達）
    print(f"学習回数: {count}回, 未到達（50ステップ経過）")
