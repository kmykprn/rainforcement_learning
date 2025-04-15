import random


def policy_random(action_list) -> list[str]:
    """
    各行動候補の行動確率をランダムに設定する

    Returns:
        各行動ごとの、選択される確率(list[str])
    """
    action_probs = [random.random() for _ in action_list]
    return action_probs


def policy_select_max_value(action_list) -> list[str]:
    """
    各行動候補の行動確率を、最も価値が高いものを1に、それ以外を0に設定する。

    Returns:
        各行動ごとの、選択される確率(list[str])
    """
    action_probs = [0] * len(action_list)
    # 行動ごとに価値を求める

    # 価値が最も高い行動を1, それ以外を0にした配列を返す

    return action_probs