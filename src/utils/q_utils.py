from typing import Tuple, Dict


def get_max_q_action(current_state: Tuple, Q: Dict[Tuple, Dict[str, float]]) -> str:
    """
    現在の状態における、価値最大の行動を取得する

    Args:
        current_state:
            現在の状態

        Q:
            Qテーブル

    Returns:
        action:
            行動
    """
    # 価値最大の行動を選択
    max_value = max(Q[current_state].values())
    action: str = [
        action for action, value in Q[current_state].items() if value == max_value
    ][0]

    return action
