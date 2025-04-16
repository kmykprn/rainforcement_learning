def V(t: int, T: int) -> int | float:
    """
    現在の時刻（ステップ）における報酬の総和を計算する

    Args:
        t:
            現在の時刻(ステップ)
        T:
            最終時刻

    Returns:
        価値
    """
    # 割引率
    gamma = 0.99
    
    # 
    