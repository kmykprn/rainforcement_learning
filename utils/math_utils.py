import math
from typing import List


def softmax(X: List[float]) -> List[float]:
    """
    softmax関数。

    Args:
        X:
            floatが格納されたリスト。

    Returns:
        softmax関数で処理されたリスト。
    """
    # リスト内の最大値を取得
    max_x = max(X)

    # 値の大小関係を維持したまま正の値に変換(数値の安定化のため、最大値を引く)
    X_exp: List[float] = [math.exp(x - max_x) for x in X]

    # 分母を算出
    total = sum(X_exp)

    return [x / total for x in X_exp]
