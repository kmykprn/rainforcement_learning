"""
モンテカルロ法の実装
"""

import random
import pickle

from common.env import EnvRanks
from common.policy import Policy


from typing import List, Tuple, Dict, TypedDict


class Experience(TypedDict):
    state: Tuple[int, int]
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


def calculate_G(rewards: List[int], gamma: float = 0.99) -> List[float]:
    """
    即時報酬の総和を求める関数

    Args:
        rewards:
            エピソード終了までの即時報酬のリスト。
            例えばrewards[t]には、s(t)ではなくs(t+1)の即時報酬が格納されている。

    Returns:
        G:
            即時報酬の総和
    """

    # T: エピソードの長さ（数式中の T に対応）
    T: int = len(rewards)
    G: List[float] = [0.0] * T
    for t in range(T):
        g = 0.0
        for k in range(T - t):  # ← k = 0 〜 T - t - 1
            g += pow(gamma, k) * rewards[t + k]
        G[t] = g
    return G


def update_Q_montecarlo(
    experiences: List[Experience],
    Q: Dict[Tuple[int, int], Dict[str, float]],
    N: Dict[Tuple[Tuple[int, int], str], int],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    「ある状態（座標）である行動（移動）をした」ときの、評価値を推定

    Args:
        experiences:
            エピソード終了までの状態, 行動, 報酬を格納したリスト。

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

    # 各状態における, 1時刻後の即時報酬のリストを取得
    rewards = [e["reward"] for e in experiences]

    # 報酬の総和を計算
    G = calculate_G(rewards)

    # 状態, 行動のペアごとに、評価値を算出
    for i, x in enumerate(experiences):
        s, a = x["state"], x["action"]

        # alphaを定義。1 /  状態sで行動aを取った回数。
        # 報酬の期待値（平均値）を求めるために使用。逐次平均を取る。
        sa = (s, a)
        N[sa] += 1
        alpha = 1 / N[sa]

        # 評価値を算出
        Q[s][a] = Q[s][a] + alpha * (G[i] - Q[s][a])

    return Q


def evaluate_Q(count: int, env: EnvRanks, Q: Dict[Tuple[int, int], Dict[str, float]]):
    """
    モンテカルロ法で学習されたQ値の、成功/失敗を判定する関数

    Args:
        count:
            Q値の学習回数
        env:
            環境
        Q:
            Qテーブル
    """

    # スタート地点
    state: Tuple[int, int] = (1, 1)

    # 価値最大の行動を行なっても、ゴールに到達しない場合があるので、50試行で打ち止め
    for _ in range(50):

        # 価値最大の行動を選択
        action: str = max(Q[state], key=Q[state].get)
        next_state: Tuple[int, int] = get_new_state(state, action)
        r: int = env.reward_func(next_state)

        # ゴールに到達した場合
        if r == GOAL_REWARD:
            print(f"学習回数: {count}回, 成功!")
            return
        # 壁で停止した場合
        elif r == WALL_REWARD:
            print(f"学習回数: {count}回, 失敗...")
            return
        # 進む
        else:
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

        # 辿ってきた状態, 行動, 即時報酬を格納するリスト
        experiences: List[Experience] = []

        # モンテカルロ法のメインループ
        while True:

            # ポリシー(epsilon-greedy法)に基づき行動確率を取得
            action_probs: List[float] = policy.epsilon_greedy(
                current_state, actions, Q, epsilon=0.5
            )

            # 行動確率に基づき、行動を選択
            action = random.choices(actions, k=1, weights=action_probs)[0]

            # 行動に基づき、次の状態を選択
            new_state: Tuple[int, int] = get_new_state(current_state, action)

            # 1時刻後の即時報酬を獲得
            r: int = env.reward_func(new_state)

            # ゴール（=即時報酬が10の地点）の場合、現在の状態, 行動, 報酬を保存してループを抜ける
            if r == GOAL_REWARD:
                experiences.append(
                    {"state": current_state, "action": action, "reward": r}
                )
                break
            # 壁（即時報酬が-2の地点）の場合、座標は更新しない
            elif r == WALL_REWARD:
                continue
            # ゴールでも壁でもない場合は、状態, 行動, 報酬を記録し、場所を更新
            else:
                experiences.append(
                    {"state": current_state, "action": action, "reward": r}
                )
                current_state = new_state

        # 経験をもとに、Qテーブルを更新
        Q = update_Q_montecarlo(experiences, Q, N)

        # 評価
        evaluate_Q(count, env, Q)

    # QとNを保存
    with open("02_MonteCarlo/q_and_n.pkl", "wb") as f:
        pickle.dump((Q, N), f)


if __name__ == "__main__":

    ENV = EnvRanks()  # 環境を定義
    ACTIONS: List[str] = ["up", "down", "left", "right"]  # 行動リストを定義

    MAX_EPISODE_SIZE = 30  # エピソードの回数（=学習回数）を定義
    GOAL_REWARD = 10
    WALL_REWARD = -2

    # Q値を初期化
    Q: Dict[Tuple[int, int], Dict[str, float]] = initialize_Q()

    # N(ある状態・行動における行動回数を記録)を初期化
    N: Dict[Tuple[Tuple[int, int], str], int] = initialize_N()

    # モンテカルロ法の実施
    main(env=ENV, actions=ACTIONS, Q=Q, N=N)
