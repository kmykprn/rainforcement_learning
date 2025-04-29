"""
Q-learningの実装
"""

import random
import torch
import yaml


from rl.core.env import EnvRanks
from rl.core.policy import Policy
from rl.core.dynamics import Ranks
from dqn.types.statictype import ExperienceDQN
from dqn.core.train import train_rl
from dqn.utils.evaluate import evaluate_model
from dqn.core.replaybuffer import ReplayBuffer
from dqn.models.simpleDQN import SimpleDQN
from typing import List, Tuple


def estimate_q_all_actions(
    current_state: Tuple[int, int],
    model: SimpleDQN,
    device: torch.device,
) -> List[float]:
    """
    現在の状態における、すべての行動に対するq値を算出する関数

    Args:
        current_state: 現在の状態（座標）
        model: DQNのモデル
        device: cude or cpu

    Returns:
        q値（リスト）
    """
    # Tupleをテンソルに変換
    current_state_tensor = torch.tensor(current_state, dtype=torch.float32).to(device)

    # モデルに入力できるように次元数を増やす (入力次元数,) -> (1, 入力次元数)
    # unsqueeze(0) で先頭に次元を追加
    current_state_tensor = current_state_tensor.unsqueeze(0)

    # モデルを評価モードに切り替え (Dropoutなどが無効になる)
    model.eval()

    # 勾配計算を無効化 (メモリ節約&高速化)
    with torch.no_grad():
        # 現在の状態における、すべての行動に対するq値を算出
        q_values: torch.Tensor = model(current_state_tensor)

    # q値をリストに変換
    q_values_list: List[float] = q_values.squeeze(0).tolist()

    return q_values_list


def main(
    model: SimpleDQN,
    target_model: SimpleDQN,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
):

    policy = Policy()
    replay_buffer = ReplayBuffer(capacity=10000)

    # 現在の状態と行動をもとに、次の状態を計算するオブジェクトを呼び出し
    rank_dynamics = Ranks()

    # ターゲットネットワークの重みを初期化 (学習開始時)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()  # ターゲットはずっと評価モード

    # 全体のステップ数をカウント (ターゲットネットワーク更新などに使う)
    global_step_count = 0

    # 学習回数をカウント(評価に使用する)
    count = 0

    for _ in range(MAX_EPISODE_SIZE):

        # 初期位置を定義(要素が0から始めると、envの範囲外を指定する場合があるので変更しない)
        current_state: Tuple[int, int] = (1, 1)

        # DeepQlearningのメインループ(MAX_STEPまでにゴールに到達しなければ打ち切り)
        for _ in range(MAX_STEP):

            global_step_count += 1

            # 現在の状態における、すべての行動に対するq値を算出
            q_values: List[float] = estimate_q_all_actions(current_state, model, device)

            # ポリシー(epsilon-greedy法)に基づきQ値を行動確率に変換
            action_probs: List[float] = policy.epsilon_greedy(q_values)

            # 行動確率に基づき、行動を選択
            action = random.choices(ACTIONS, k=1, weights=action_probs)[0]

            # 行動に基づき、次の状態を選択
            new_state: Tuple[int, int] = rank_dynamics.get_new_state(
                current_state, action
            )

            # 即時報酬を獲得
            reward: int = ENV.reward_func(new_state)

            # 経験を成形(次の状態がゴールならdoneはTrue, それ以外はFalse)
            experience: ExperienceDQN = {
                "state": current_state,
                "action": action,
                "reward": reward,
                "next_state": new_state,
                "done": (reward == GOAL_REWARD),
            }

            # バッファに保存
            replay_buffer.add(experience)

            # ゴールの場合、現在の状態, 行動, 次の状態, 即時報酬, エピソードの終了状態を格納してループを抜ける
            if reward == GOAL_REWARD:
                break
            # 壁（即時報酬が-2の地点）の場合、座標は更新しない
            elif reward == WALL_REWARD:
                continue
            # 状態を遷移
            else:
                current_state = new_state

            # モデルを学習
            if len(replay_buffer) >= BATCH_SIZE:
                count += 1
                train_rl(
                    device=device,
                    model=model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    actions_list=ACTIONS,
                    batch_size=BATCH_SIZE,
                    gamma=0.99,
                )

            # ターゲットネットワーク更新 (一定ステップごと)
            if global_step_count % TARGET_UPDATE_FREQ == 0:
                target_model.load_state_dict(model.state_dict())

        # 評価
        evaluate_model(
            count=count,
            env=ENV,
            actions_list=ACTIONS,
            model=model,
            device=device,
            goal_reward=GOAL_REWARD,
            wall_reward=WALL_REWARD,
        )


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

    # エピソードの回数（=学習回数）を定義
    MAX_EPISODE_SIZE: int = config["learning"]["max_episode_size"]

    # 試行の打ち切り回数を定義
    MAX_STEP: int = config["learning"]["max_step"]

    # バッチサイズを定義
    BATCH_SIZE: int = 32

    # ターゲットモデルの更新頻度を定義
    TARGET_UPDATE_FREQ = 10

    # gpuかcpuを使用。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NNモデルを別ファイルから呼び出し
    model = SimpleDQN(input_dim=2, output_dim=len(ACTIONS)).to(device)
    target_model = SimpleDQN(input_dim=2, output_dim=len(ACTIONS)).to(device)

    # 重みの読み込み元 兼 保存先を定義
    WEIGHTS_PATH = "weights/simple_dqn.pth"

    # 重みをロード
    # if os.path.exists(WEIGHTS_PATH):
    #    model.load_state_dict(torch.load(WEIGHTS_PATH))

    # 学習率を定義
    learning_rate = 1e-3

    # optimizer(SGD: θ-lr * Δθ)の定義
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DeepQlearningの実施
    main(model=model, target_model=target_model, device=device, optimizer=optimizer)

    # 重みを保存
    torch.save(model.state_dict(), WEIGHTS_PATH)
