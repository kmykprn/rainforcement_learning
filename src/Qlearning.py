"""
Q-learningの実装
"""

import random
import pickle
import yaml


from rl.core.env import EnvRanks
from rl.core.policy import Policy
from rl.core.dynamics import Ranks
from rl.utils.evaluate import evaluate_Q
from rl.utils.initializer import initialize_Q, initialize_N
from typing import List, Tuple, Dict, TypedDict


class ExperienceQL(TypedDict):
    """
    Qlearning用の静的タイプ
    """

    state: Tuple[int, int]
    new_state: Tuple[int, int]
    action: str
    reward: int


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

        # 現在の状態と行動をもとに、次の状態を計算するオブジェクトを呼び出し
        rank_dynamics = Ranks()

        # Q-learningのメインループ(MAX_STEPまでにゴールに到達しなければ打ち切り)
        for _ in range(MAX_STEP):

            # 現在の状態における, 行動ごとのQ値を取り出し
            q_values: List[float] = list(Q[current_state].values())

            # ポリシー(epsilon-greedy法)に基づきQ値を行動確率に変換
            action_probs: List[float] = policy.epsilon_greedy(q_values)

            # 行動確率に基づき、行動を選択
            action = random.choices(actions, k=1, weights=action_probs)[0]

            # 行動に基づき、次の状態を選択
            new_state: Tuple[int, int] = rank_dynamics.get_new_state(
                current_state, action
            )

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
        evaluate_Q(count, env, Q, GOAL_REWARD, WALL_REWARD)

    # QとNを保存
    with open("weights/Qlearning.pkl", "wb") as f:
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

    # Qlearningの実施
    main(env=ENV, actions=ACTIONS, Q=Q, N=N)
