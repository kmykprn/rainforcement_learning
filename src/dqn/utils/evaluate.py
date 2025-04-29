from rl.core.env import EnvRanks
from rl.core.dynamics import Ranks
from src.dqn.models.simpleDQN import SimpleDQN

import torch
from typing import List, Tuple


def get_max_nn_action(
    current_state: Tuple,
    model: SimpleDQN,
    device: torch.device,
    actions_list: List[str],
) -> str:
    """
    現在の状態における、価値最大の行動を取得する

    Args:
        current_state: 現在の状態
        model: DQNモデル

    Returns:
        action: 行動
    """

    model.eval()

    # 勾配計算は不要
    with torch.no_grad():

        # Tupleをテンソルに変換し、モデルと同じdeviceへ格納
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32).to(
            device
        )

        # モデルに入力できるように、バッチサイズ用の次元を追加
        current_state_tensor = current_state_tensor.unsqueeze(0)

        # すべての行動における価値（Q[current_state]）を推論
        q_values = model(current_state_tensor)

        # Q値最大の行動を取得。
        # torch.argmax(q_values, dim=1) で、各行の最大値のインデックスを取得。
        # バッチサイズは1だから、結果は要素1つのテンソル。 .item() でテンソル内の数値を取得
        max_action_index: int = int(torch.argmax(q_values, dim=1).item())
        action: str = actions_list[max_action_index]

    return action


def evaluate_model(
    count: int,
    env: EnvRanks,
    actions_list: List[str],
    model: SimpleDQN,
    device: torch.device,
    goal_reward: int,
    wall_reward: int,
):
    """
    学習済みNNモデルで、ゴール到達成功/失敗を判定する関数

    Args:
        count: NNの学習回数
        env: 環境
        model: nnモデル
        goal_reward: ゴールに到達したときの即時報酬
        wall_reward: カベに到達したときの即時報酬
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
        action: str = get_max_nn_action(state, model, device, actions_list)
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
