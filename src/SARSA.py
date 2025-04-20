"""
SARSAの実装
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


class ExperienceSARSA(TypedDict):
    """
    SARSA用の静的タイプ
    """

    state: Tuple[int, int]
    action: str
    reward: int
    new_state: Tuple[int, int]
    new_action: str


def update_Q_SARSA(
    experience: ExperienceSARSA,
    Q: Dict[Tuple[int, int], Dict[str, float]],
    N: Dict[Tuple[Tuple[int, int], str], int],
    gamma: float = 0.9,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    「ある状態（座標）である行動（移動）をした」ときの、評価値を推定

    Args:
        experience:
            現在の状態（座標）, 行動, 次の状態, 次の行動, 報酬が格納された辞書

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

    s, a, r, s_new, a_new = (
        experience["state"],
        experience["action"],
        experience["reward"],
        experience["new_state"],
        experience["new_action"],
    )

    # alphaを定義。1 /  状態sで行動aを取った回数。
    # 報酬の期待値（平均値）を求めるために使用。逐次平均を取る。
    sa = (s, a)
    N[sa] += 1
    alpha = 1 / N[sa]

    # Qテーブルを更新
    Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s_new][a_new] - Q[s][a])

    return Q


def main(
    env: EnvRanks,
    actions: List[str],
    Q: Dict[Tuple[int, int], Dict[str, float]],
    N: Dict[Tuple[Tuple[int, int], str], int],
):

    policy = Policy()

    for count in range(MAX_EPISODE_SIZE):

        """
        初期化
        """
        # 初期位置を定義(要素が0から始めると、envの範囲外を指定する場合があるので変更しない)
        current_state: Tuple[int, int] = (1, 1)

        # ポリシー(epsilon-greedy法)に基づき行動確率を取得
        action_probs: List[float] = policy.epsilon_greedy(
            current_state, actions, Q, epsilon=0.5
        )

        # 行動確率に基づき、行動を選択
        action = random.choices(actions, k=1, weights=action_probs)[0]

        # 現在の状態と行動をもとに、次の状態を計算するオブジェクトを呼び出し
        rank_dynamics = Ranks()

        # SARSAのメインループ(MAX_STEPまでにゴールに到達しなければ打ち切り)
        for _ in range(MAX_STEP):

            """
            次の状態における価値（実測値）を計算
            """
            # 行動に基づき、次の状態を選択
            new_state: Tuple[int, int] = rank_dynamics.get_new_state(
                current_state, action
            )

            # 1時刻後の即時報酬を獲得
            r_new_state: int = env.reward_func(new_state)

            # ポリシーに基づき、次の状態における行動確率を取得
            new_action_probs: List[float] = policy.epsilon_greedy(
                new_state, actions, Q, epsilon=0.5
            )

            # 行動確率に基づき、行動を選択
            new_action = random.choices(actions, k=1, weights=new_action_probs)[0]

            # 現在の状態, 行動, 次の状態, 即時報酬を格納
            experience: ExperienceSARSA = {
                "state": current_state,
                "action": action,
                "reward": r_new_state,
                "new_state": new_state,
                "new_action": new_action,
            }

            # ゴールの場合、Qテーブルを更新してループを抜ける
            if r_new_state == GOAL_REWARD:
                update_Q_SARSA(experience, Q, N)
                break
            # 壁（即時報酬が-2の地点）の場合、行動のみ取り直し、状態は更新しない
            elif r_new_state == WALL_REWARD:
                action_probs: List[float] = policy.epsilon_greedy(
                    current_state, actions, Q, epsilon=0.5
                )
                action = random.choices(actions, k=1, weights=action_probs)[0]
                continue
            # 現在の状態とQテーブルを更新
            else:
                update_Q_SARSA(experience, Q, N)
                current_state = new_state
                action = new_action

        # 評価
        evaluate_Q(count, env, Q, GOAL_REWARD, WALL_REWARD)

    # QとNを保存
    with open("weights/SARSA.pkl", "wb") as f:
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
