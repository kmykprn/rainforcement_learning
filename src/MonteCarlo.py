"""
モンテカルロ法の実装
"""

import random
import pickle
import yaml

from core.env import EnvRanks
from core.policy import Policy
from core.dynamics import Ranks
from utils.evaluate import evaluate_Q
from utils.initializer import initialize_Q, initialize_N
from typing import List, Tuple, Dict, TypedDict


class Experience(TypedDict):
    state: Tuple[int, int]
    action: str
    reward: int


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


def main(
    env: EnvRanks,
    actions: List[str],
    Q: Dict[Tuple[int, int], Dict[str, float]],
    N: Dict[Tuple[Tuple[int, int], str], int],
):

    policy = Policy()

    # 現在の状態と行動をもとに、次の状態を計算するオブジェクトを呼び出し
    rank_dynamics = Ranks()

    for count in range(MAX_EPISODE_SIZE):

        # 初期位置を定義(要素が0から始めると、envの範囲外を指定する場合があるので変更しない)
        current_state: Tuple[int, int] = (1, 1)

        # 辿ってきた状態, 行動, 即時報酬を格納するリスト
        experiences: List[Experience] = []

        # モンテカルロ法のメインループ(MAX_STEPまでにゴールに到達しなければ打ち切り)
        for _ in range(MAX_STEP):

            # ポリシー(epsilon-greedy法)に基づき行動確率を取得
            action_probs: List[float] = policy.epsilon_greedy(
                current_state, actions, Q, epsilon=0.5
            )

            # 行動確率に基づき、行動を選択
            action = random.choices(actions, k=1, weights=action_probs)[0]

            # 行動に基づき、次の状態を選択
            new_state: Tuple[int, int] = rank_dynamics.get_new_state(
                current_state, action
            )

            # 1時刻後の即時報酬を獲得
            r_new_state: int = env.reward_func(new_state)

            # ゴール（=即時報酬が10の地点）の場合、現在の状態, 行動, 報酬を保存してループを抜ける
            if r_new_state == GOAL_REWARD:
                experiences.append(
                    {"state": current_state, "action": action, "reward": r_new_state}
                )
                break
            # 壁（即時報酬が-2の地点）の場合、座標は更新しない
            elif r_new_state == WALL_REWARD:
                continue
            # ゴールでも壁でもない場合は、状態, 行動, 報酬を記録し、場所を更新
            else:
                experiences.append(
                    {"state": current_state, "action": action, "reward": r_new_state}
                )
                current_state = new_state

        # 経験をもとに、Qテーブルを更新
        Q = update_Q_montecarlo(experiences, Q, N)

        # 評価
        evaluate_Q(count, env, Q, GOAL_REWARD, WALL_REWARD)

    # QとNを保存
    with open("weights/MonteCarlo.pkl", "wb") as f:
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

    # 環境の全状態(今回は座標)をリストに格納
    states: List[Tuple[int, int]] = ENV.get_states()

    # Q値を初期化
    Q: Dict[Tuple[int, int], Dict[str, float]] = initialize_Q(states, ACTIONS, "zeros")

    # N(ある状態・行動における行動回数を記録)を初期化
    N: Dict[Tuple[Tuple[int, int], str], int] = initialize_N(states, ACTIONS)

    # モンテカルロ法の実施
    main(env=ENV, actions=ACTIONS, Q=Q, N=N)
