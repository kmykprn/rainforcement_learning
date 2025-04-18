from typing import Tuple, Dict, List


def initialize_Q(
    states: List[Tuple],
    actions: List[str],
    initial_value: float = 0.0,
) -> Dict[Tuple, Dict[str, float]]:
    """
    全状態（座標）における、すべての行動確率を初期化する。
    状態の次元数は任意に変更できるよう、Tuple を使用して表現。

    Args:
        states:
            すべての状態を持つオブジェクト。

        actions:
            すべての行動を持つオブジェクト。

        initial_value:
            Qの初期値。

    Returns:
        initial_Q:
            初期値設定後のQテーブル。
            例. {
                    (1, 1): {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0},
                    ...
                }
    """

    initial_Q: Dict[Tuple, Dict[str, float]] = {}
    for state in states:

        # 例. action_probs = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}
        action_probs = {action: initial_value for action in actions}

        # Qテーブルを初期化
        initial_Q[state] = action_probs

    return initial_Q


def initialize_N(
    states: List[Tuple],
    actions: List[str],
) -> Dict[Tuple[Tuple, str], int]:
    """
    全状態（座標）における、すべての行動回数を初期化する。
    状態の次元数は任意に変更できるよう、Tuple を使用して表現。

    Args:
        states:
            すべての状態を持つオブジェクト。

        actions:
            すべての行動を持つオブジェクト。

    Returns:
        initial_N:
            初期値設定後のN。
            例. {
                    ( (1, 1), 'up'): 5,
                    ...
                }
    """

    # 全状態（座標）における、すべての行動回数を0で初期化
    initial_N: Dict[Tuple[Tuple, str], int] = {}
    for state in states:
        for action in actions:
            sa = (state, action)
            initial_N[sa] = 0

    return initial_N
