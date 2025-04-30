import torch
import yaml
from typing import List
from src.dqn.models.simpleDQN import SimpleDQN
from src.dqn.types.statictype import ExperienceDQN
from src.dqn.core.train import experiences_to_tensor


def test_tensor():
    t = torch.Tensor([1, 2, 3])

    # 外側に括弧を一つ増やす
    t = t.unsqueeze(0)
    print("0に次元数を追加")
    print(t)  # tensor([[1., 2., 3.]])

    # 外側に括弧を一つ増やす
    t = t.unsqueeze(0)
    print("0に次元数を追加")
    print(t)  # tensor([[[1., 2., 3.]]])

    t2 = torch.Tensor([1, 2, 3])

    # 要素1つずつに括弧をつける
    t2 = t2.unsqueeze(1)
    print("1に次元数を追加")
    print(t2)  # tensor([[1.], [2.], [3.]])

    # 要素1つずつに括弧をつける
    t2 = t2.unsqueeze(1)
    print("1に次元数を追加")
    print(t2)  # tensor([[[1.]], [[2.]], [[3.]]])

    t3 = torch.Tensor([[1, 2, 3], [4, 5, 6]])

    # 外側に括弧をつける
    t3_0 = t3.unsqueeze(0)
    print("0に次元数を追加")
    print(t3_0)  # tensor([[1, 2, 3], [4, 5, 6]])
    print(t3_0.shape)

    # 要素1つずつに括弧をつける
    t3_1 = t3.unsqueeze(1)
    print("1に次元数を追加")
    print(t3_1)  # tensor([[[1, 2, 3]], [[4, 5, 6]]])
    print(t3_1.shape)

    # 要素内の要素1つずつに括弧をつける
    t3_2 = t3.unsqueeze(2)
    print("2に次元数を追加")
    print(t3_2)  # tensor([[[1], [2], [3]]], [[[4], [5], [6]]]])
    print(t3_2.shape)


def test_model(model: SimpleDQN, states: torch.Tensor) -> None:
    """
    DQNモデルへの入出力を確認するための関数

    出力値の例: 各状態に対し、行動確率（Q値）が出力される
        tensor(
            [
                [-0.0801,  0.2743, -0.2159, -0.0567], # 状態1の行動確率
                [-0.1170,  0.5293, -0.1881, -0.0054], # 状態2の行動確率
                [-0.1662,  0.7667, -0.1642,  0.0643]  # 状態3の行動確率
            ], device='cuda:0', grad_fn=<AddmmBackward0>)
    """
    print(model(states))


if __name__ == "__main__":
    experience1: ExperienceDQN = {
        "state": (1, 1),
        "action": "up",
        "reward": 10,
        "next_state": (1, 2),
        "done": False,
    }

    experience2: ExperienceDQN = {
        "state": (1, 2),
        "action": "up",
        "reward": 15,
        "next_state": (1, 3),
        "done": False,
    }

    experience3: ExperienceDQN = {
        "state": (1, 3),
        "action": "right",
        "reward": 20,
        "next_state": (2, 3),
        "done": True,
    }

    experiences = [experience1, experience2, experience3]

    # gpuかcpuを使用。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 設定ファイルを読み込み
    with open("config/base.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 行動リストを定義
    ACTIONS: List[str] = config["actions"]

    # モデルを定義
    model = SimpleDQN().to(device)

    # 状態のtensorを取得
    states, action_indices, rewads, next_states, dones = experiences_to_tensor(
        experiences, ACTIONS, device
    )
    # test_model(model, states)

    # print(action_indices)
    # print(action_indices.unsqueeze(1))
    test_tensor()
