import random
from typing import List
from collections import deque
from src.dqn.types.statictype import ExperienceDQN


class ReplayBuffer:
    """
    経験をためるためのバッファ
    """

    def __init__(self, capacity: int):
        self.buffer: deque[ExperienceDQN] = deque(maxlen=capacity)

    def add(self, experience: ExperienceDQN) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[ExperienceDQN]:
        """
        経験をランダムにバッチサイズ分サンプルして返却する関数
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """クラスに対してlen()を使用できるようにする"""
        return len(self.buffer)
