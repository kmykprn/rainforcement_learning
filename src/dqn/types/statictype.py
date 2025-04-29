from typing import Tuple, TypedDict


class ExperienceDQN(TypedDict):
    """
    DeepQlearning用の静的タイプ
    """

    state: Tuple[int, int]
    action: str
    reward: int
    next_state: Tuple[int, int]
    done: bool
