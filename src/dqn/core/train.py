import torch
from typing import List, Tuple
from src.dqn.types.statictype import ExperienceDQN
from src.dqn.core.replaybuffer import ReplayBuffer
from src.dqn.models.simpleDQN import SimpleDQN
from torch.utils.tensorboard.writer import SummaryWriter


def experiences_to_tensor(
    experiences: List[ExperienceDQN], actions_list: List[str], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ReplayBufferからサンプリングした経験をtensorに変換する関数

    Args:
        sampled_experiences: ReplayBufferからサンプリングされた経験
        actions: 取りうる行動のリスト (例: ["up", "down", "left", "right"])
        device: デバイス情報

    Returns:
        states: 状態を格納したテンソル
            例: tensor([[1., 1.], [1., 2.], [1., 3.]], device='cuda:0')

        action_indices: 行動をインデックス化し、格納したテンソル
            例: tensor([0, 0, 3], device='cuda:0')

        rewards: 即時報酬を格納したテンソル
            例: tensor([10., 15., 20.], device='cuda:0')

        next_states: 次の状態を格納したテンソル
            例: tensor([[1., 2.], [1., 3.], [2., 3.]], device='cuda:0')

        dones: next_statesがゴールか否かをTrue/Falseで格納したテンソル
            例: tensor([0., 0., 1.], device='cuda:0')
    """

    # statesの例. tensor([[1., 1.], [1., 2.], [1., 3.]], device='cuda:0')
    states = torch.tensor([e["state"] for e in experiences], dtype=torch.float32).to(
        device
    )

    # 行動はstrなので、インデックス値に変換してからtensor化
    # actionsの例. tensor([0, 0, 3], device='cuda:0')
    action_indices = torch.tensor(
        [actions_list.index(e["action"]) for e in experiences], dtype=torch.int64
    ).to(device)

    # rewardsの例. tensor([10., 15., 20.], device='cuda:0')
    # (batch_size, 1) になる
    rewards = (
        torch.tensor([e["reward"] for e in experiences], dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    # next_statesの例. tensor([[1., 2.], [1., 3.], [2., 3.]], device='cuda:0')
    next_states = torch.tensor(
        [e["next_state"] for e in experiences], dtype=torch.float32
    ).to(device)

    # donesの例. tensor([0., 0., 1.], device='cuda:0')
    dones = torch.tensor([e["done"] for e in experiences], dtype=torch.float32).to(
        device
    )

    return states, action_indices, rewards, next_states, dones


def estimate_q_values(
    current_states: torch.Tensor,
    action_indices: torch.Tensor,
    model: SimpleDQN,
) -> torch.Tensor:
    """
    現在の状態, 取った行動における価値（Q[state][action]）を推論する関数

    Args:
        current_states: 現在の状態を格納したテンソル
            例: tensor([[1., 1.], [1., 2.], [1., 3.]], device='cuda:0')

        action_indices: 実際に取った行動をインデックス化し、格納したテンソル
            例: tensor([0, 0, 3], device='cuda:0')

        model: Q値を推論するためのモデル
            出力例. tensor(
            [
                [-0.0801,  0.2743, -0.2159, -0.0567], # 状態1の行動確率
                [-0.1170,  0.5293, -0.1881, -0.0054], # 状態2の行動確率
                [-0.1662,  0.7667, -0.1642,  0.0643]  # 状態3の行動確率
            ], device='cuda:0', grad_fn=<AddmmBackward0>)

    Returns:
        torch.Tensor:  現在の状態, 取った行動における価値（Q[state][action]）
            例. tensor([-0.0801, -0.1170, 0.0643])
    """

    # 状態ごとに、すべての行動における価値（Q[state]）を推論
    q_values = model(current_states)

    # 実際に取った行動における価値（Q[state][action]）のみ抽出
    selected_q_values_list = [q[a] for q, a in zip(q_values, action_indices)]

    # torch.stack でリスト内のテンソルを結合
    # これにより、計算グラフが維持されたまま1次元テンソル (batch_size,) が得られる
    estimated_q = torch.stack(selected_q_values_list)

    # 出力は (batch_size,) の形状なので、(batch_size, 1)に変換
    estimated_q = estimated_q.unsqueeze(1)

    # 1次元のテンソルに変換
    return estimated_q


def estimate_next_q_max_values(
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_model: SimpleDQN,
    device: torch.device,
) -> torch.Tensor:
    """
    次の状態における価値最大の行動のQ値（Q[state][max(action)]）を推論する関数

    Args:
        next_states: 次の状態を格納したテンソル
            例: tensor([[1., 1.], [1., 2.], [1., 3.]], device='cuda:0')

        dones: next_statesがゴールか否かをTrue/Falseで格納したテンソル
            例: tensor([0., 0., 1.], device='cuda:0')

        target_model: Q値を推論するためのターゲットモデル
            出力例. tensor(
            [
                [-0.0801,  0.2743, -0.2159, -0.0567], # 状態1の行動確率
                [-0.1170,  0.5293, -0.1881, -0.0054], # 状態2の行動確率
                [-0.1662,  0.7667, -0.1642,  0.0643]  # 状態3の行動確率
            ], device='cuda:0', grad_fn=<AddmmBackward0>)

        device: gpu or cpu

    Returns:
        torch.Tensor:  次の状態における価値最大の行動のQ値（Q[state][max(action)]）。
        勾配は計算されない。
            例. tensor([0.2743, 0.5293, 0.7667])
    """

    # ターゲットネットワークの計算では勾配は不要
    with torch.no_grad():

        # 状態ごとに、すべての行動における価値（Q[state]）を推論
        next_q_values = target_model(next_states)

        # 価値最大の行動（Q[next_state][max(action)]）のみ抽出
        next_q_values_max_list = [max(q) for q in next_q_values]

        # next_statesがゴールの場合、以降は価値が得られないため行動確率(Q値)を0にする
        for i, done in enumerate(dones):
            if done:
                next_q_values_max_list[i] = 0.0

        # 戻り値をtorch.tensor(batch, 1)で返す
        next_q_values_max = torch.tensor(
            next_q_values_max_list, dtype=torch.float32, device=device
        ).unsqueeze(1)

    # 1次元のテンソルに変換
    return next_q_values_max


def estimate_next_q_double_dqn(
    next_states: torch.Tensor,
    dones: torch.Tensor,
    model: SimpleDQN,  # オンラインネットワーク
    target_model: SimpleDQN,  # ターゲットネットワーク
    device: torch.device,
) -> torch.Tensor:
    """
    Double DQN用の次の状態における価値最大の行動のQ値を推論する関数

    オンラインネットワークで行動を選択し、ターゲットネットワークでその行動の価値を評価する

    Args:
        next_states: 次の状態を格納したテンソル
            例: tensor([[1., 1.], [1., 2.], [1., 3.]], device='cuda:0')

        dones: next_statesがゴールか否かをTrue/Falseで格納したテンソル
            例: tensor([0., 0., 1.], device='cuda:0')

        model: 行動選択用のオンラインネットワーク

        target_model: 行動評価用のターゲットネットワーク

        device: gpu or cpu

    Returns:
        torch.Tensor: Double DQNで計算した次の状態における価値
    """

    # 勾配計算は不要
    with torch.no_grad():
        # オンラインネットワークで次の状態での行動の価値を取得(shape: batch, 4)
        next_q_values_online = model(next_states)

        # 次の状態における行動選択をオンラインネットワークで実施。
        # 各状態での最大Q値を持つ行動のインデックスを取得(shape: batch, 1)
        # dim=1で、行動の次元を見てargmaxを取得
        best_actions = next_q_values_online.argmax(dim=1, keepdim=True)

        # ターゲットネットワークで、オンラインネットワークにより選択された状態, 行動の価値を取得(shape: batch, 4)
        next_q_values_target = target_model(next_states)

        # ターゲットネットワークで、次の状態でベストアクションの価値を取得
        # .gather()は、dimで指定した軸（今回は行動の次元）に対し、indexで指定した行動を取得
        # なぜこれで安定性が確保できるかモヤモヤは残るが、一旦手法自体は理解したので先に進む。
        next_q_values = next_q_values_target.gather(
            dim=1, index=best_actions
        )  # shape: (batch, 1)

        # オンラインネットワークで選択された行動の価値を取得
        next_q_values_list = []
        for i, action_idx in enumerate(best_actions):
            next_q_values_list.append(next_q_values_target[i][action_idx].item())

        # next_statesがゴールの場合、以降は価値が得られないため行動確率(Q値)を0にする
        for i, done in enumerate(dones):
            if done:
                next_q_values_list[i] = 0.0

        # 戻り値をtorch.tensor(batch, 1)で返す
        next_q_values = torch.tensor(
            next_q_values_list, dtype=torch.float32, device=device
        ).unsqueeze(1)

    return next_q_values


def train_rl(
    device: torch.device,
    model: SimpleDQN,
    target_model: SimpleDQN,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    actions_list: List[str],  # actions を引数で受け取る
    batch_size: int,
    writer: SummaryWriter,
    gamma: float = 0.99,
    global_step: int = 0,
):
    """
    深層強化学習の学習用関数
    """

    # 十分なデータ量がたまるまで学習はしない
    if len(replay_buffer) < batch_size:
        return

    # モデルを訓練モードに設定
    model.train()

    # ReplayBufferから経験をサンプリング
    experiences: List[ExperienceDQN] = replay_buffer.sample(batch_size)

    # 経験テンソルに格納
    states, action_indices, rewards, next_states, dones = experiences_to_tensor(
        experiences, actions_list, device
    )

    # 現在の状態, 取った行動における価値（Q[state][action]）をモデルで推論
    estimated_q = estimate_q_values(states, action_indices, model)

    # 次の状態におけるQ値の中から最大の値を取り出し
    # target_modelを使用することで、TD誤差を安定化する
    next_q_values_max = estimate_next_q_max_values(
        next_states, dones, target_model, device
    )

    # 実際に得られた価値（Q値）を計算
    gain_q = rewards + gamma * next_q_values_max

    # 前回の勾配をリセット
    optimizer.zero_grad()

    # Huber損失
    loss = torch.nn.functional.smooth_l1_loss(estimated_q, gain_q)

    # 勾配を計算
    loss.backward()

    # 重みを更新
    optimizer.step()

    if writer:  # 👈 TensorBoardを使うために追加
        writer.add_scalar("Loss/train", loss.item(), global_step)

    if global_step % 100 == 0:
        loss = loss.item()
        print(f"loss: {loss:.7f} count: {global_step: 5d}")


def train_double_dqn(
    device: torch.device,
    model: SimpleDQN,
    target_model: SimpleDQN,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    actions_list: List[str],
    batch_size: int,
    writer: SummaryWriter,
    gamma: float = 0.99,
    global_step: int = 0,
):
    """
    Double DQNの学習用関数

    通常のDQNとの違いは、次の状態での行動選択と評価を別々のネットワークで行うこと
    """

    # 十分なデータ量がたまるまで学習はしない
    if len(replay_buffer) < batch_size:
        return

    # モデルを訓練モードに設定
    model.train()

    # ReplayBufferから経験をサンプリング
    experiences: List[ExperienceDQN] = replay_buffer.sample(batch_size)

    # 経験テンソルに格納
    states, action_indices, rewards, next_states, dones = experiences_to_tensor(
        experiences, actions_list, device
    )

    # 現在の状態, 取った行動における価値（Q[state][action]）をモデルで推論
    estimated_q = estimate_q_values(states, action_indices, model)

    # Double DQN: 次の状態での行動選択と評価を別々のネットワークで行う
    # オンラインネットワーク(model)で行動を選択し、ターゲットネットワーク(target_model)でその行動の価値を評価
    next_q_values = estimate_next_q_double_dqn(
        next_states, dones, model, target_model, device
    )

    # 実際に得られた価値（Q値）を計算
    gain_q = rewards + gamma * next_q_values

    # 前回の勾配をリセット
    optimizer.zero_grad()

    # Huber損失
    loss = torch.nn.functional.smooth_l1_loss(estimated_q, gain_q)

    # 勾配を計算
    loss.backward()

    # 重みを更新
    optimizer.step()

    if writer:
        writer.add_scalar("Loss/train", loss.item(), global_step)

    if global_step % 100 == 0:
        loss = loss.item()
        print(f"loss: {loss:.7f} count: {global_step: 5d}")
