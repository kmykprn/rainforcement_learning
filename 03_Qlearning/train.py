"""
Q-learningの実装
"""

import random
import pickle
import yaml


from core.env import EnvRanks
from core.policy import Policy
from typing import List, Tuple, Dict, TypedDict


class ExperienceQL(TypedDict):
    """
    Qlearning用の静的タイプ
    """

    state: Tuple[int, int]
    new_state: Tuple[int, int]
    action: str
    reward: int


def initialize_Q() -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    全状態（座標）における、すべての行動確率を初期化

    Args:
        None

    Returns:
        initial_Q:
            Qの初期値。
            例. Q = {
                        (1, 1): {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0},
                        ...
                }
    """

    # 環境の行数, 列数を取得
    rows = len(ENV.ranks)
    cols = len(ENV.ranks[0])

    # 全状態（座標）における、すべての行動確率を0.0で初期化
    initial_Q = {}
    for col in range(cols):
        for row in range(rows):
            state: Tuple[int, int] = (row, col)
            action_probs = {action: 0.0 for action in ACTIONS}
            initial_Q[state] = action_probs

    return initial_Q


def initialize_N() -> Dict[Tuple[Tuple[int, int], str], int]:
    """
    全状態（座標）における、すべての行動回数を初期化

    Args:
        None

    Returns:
        initial_N:
            Nの初期値。
            例. N = {
                        ( (1, 1), 'up'): 5,
                        ...
                }
    """

    # 環境の行数, 列数を取得
    rows: int = len(ENV.ranks)
    cols: int = len(ENV.ranks[0])

    # 全状態（座標）における、すべての行動確率を0.0で初期化
    initial_N: Dict[Tuple[Tuple[int, int], str], int] = {}
    for col in range(cols):
        for row in range(rows):
            state: Tuple[int, int] = (row, col)
            for action in ACTIONS:
                sa = (state, action)
                initial_N[sa] = 0

    return initial_N


def get_new_state(current_state: Tuple[int, int], action: str) -> Tuple[int, int]:
    """
    現在の状態と行動を受け取り、次の状態を返す関数

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


def update_Q_Qlearning(
    experience: ExperienceQL,
    Q: Dict[Tuple[int, int], Dict[str, float]],
    N: Dict[Tuple[Tuple[int, int], str], int],
    gamma: float = 0.9,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    「ある状態（座標）である行動（移動）をした」ときの、評価値を推定

    Args:
        experience:
            現在の状態（座標）, 行動, 次の状態, 報酬が格納された辞書

        Q:
            Qテーブル。
            例. Q = {
                        (1, 1): {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0},
                        ...
                }
        N:
            ある状態で, ある行動が行なわれた回数。
            例. N = {
                        ( (1, 1), 'up') : 5
                        ...
                }

    Returns:
        Q:
            行動確率が更新されたQテーブル
    """

    s, a, s_new, r = (
        experience["state"],
        experience["action"],
        experience["new_state"],
        experience["reward"],
    )

    # alphaを定義。1 /  状態sで行動aを取った回数。
    # 報酬の期待値（平均値）を求めるために使用。逐次平均を取る。
    sa = (s, a)
    N[sa] += 1
    alpha = 1 / N[sa]

    # Qテーブルを更新
    Q[s][a] = Q[s][a] + alpha * ((r + gamma * max(Q[s_new].values())) - Q[s][a])

    return Q


def evaluate_Q(count: int, env: EnvRanks, Q: Dict[Tuple[int, int], Dict[str, float]]):
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

    # 価値最大の行動を行なっても、ゴールに到達しない場合があるので、50試行で打ち止め
    for _ in range(50):

        # 価値最大の行動を選択
        action: str = max(Q[state], key=Q[state].get)
        next_state: Tuple[int, int] = get_new_state(state, action)
        r: int = env.reward_func(next_state)

        # ゴールに到達した場合
        if r == GOAL_REWARD:
            path_through.append(next_state)
            print(f"学習回数: {count}回, 成功!")
            print(f"経路：{path_through}")
            return
        # 壁で停止した場合
        elif r == WALL_REWARD:
            print(f"学習回数: {count}回, 失敗...")
            return
        # 進む
        else:
            path_through.append(next_state)
            state = next_state

    # 上記のいずれでも終了しなかった場合（未到達）
    print(f"学習回数: {count}回, 未到達（50ステップ経過）")


def main(
    env: EnvRanks,
    actions: List[str],
    Q: Dict[Tuple[int, int], Dict[str, float]],
    N: Dict[Tuple[Tuple[int, int], str], int],
):

    policy = Policy()

    for count in range(MAX_EPISODE_SIZE):

        # 初期位置を定義(要素が0から始めると、envの範囲外を指定する場合があるので変更しない)
        current_state: Tuple[int, int] = (1, 1)

        # Q-learningのメインループ(MAX_STEPまでにゴールに到達しなければ打ち切り)
        for _ in range(MAX_STEP):

            # ポリシー(epsilon-greedy法)に基づき行動確率を取得
            action_probs: List[float] = policy.epsilon_greedy(
                current_state, actions, Q, epsilon=0.5
            )

            # 行動確率に基づき、行動を選択
            action = random.choices(actions, k=1, weights=action_probs)[0]

            # 行動に基づき、次の状態を選択
            new_state: Tuple[int, int] = get_new_state(current_state, action)

            # 1時刻後の即時報酬を獲得
            r_new_state: int = env.reward_func(new_state)

            # 現在の状態, 行動, 次の状態, 即時報酬を格納
            experience: ExperienceQL = {
                "state": current_state,
                "action": action,
                "new_state": new_state,
                "reward": r_new_state,
            }

            # ゴールの場合、Qテーブルを更新してループを抜ける
            if r_new_state == GOAL_REWARD:
                update_Q_Qlearning(experience, Q, N)
                break
            # 壁（即時報酬が-2の地点）の場合、座標は更新しない
            elif r_new_state == WALL_REWARD:
                continue
            # 場所とQテーブルを更新
            else:
                update_Q_Qlearning(experience, Q, N)
                current_state = new_state

        # 評価
        evaluate_Q(count, env, Q)

    # QとNを保存
    with open("03_Qlearning/q_and_n.pkl", "wb") as f:
        pickle.dump((Q, N), f)


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

    # Q値を初期化
    Q: Dict[Tuple[int, int], Dict[str, float]] = initialize_Q()

    # N(ある状態・行動における行動回数を記録)を初期化
    N: Dict[Tuple[Tuple[int, int], str], int] = initialize_N()

    # Qlearningの実施
    main(env=ENV, actions=ACTIONS, Q=Q, N=N)
