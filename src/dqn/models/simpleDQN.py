from torch import nn


class SimpleDQN(nn.Module):
    """
    グリッド環境におけるDQNモデル。
    2次元座標(x, y)を入力とし、各行動のQ値を算出する
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 4):
        super(SimpleDQN, self).__init__()

        # 畳み込み処理部分
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_dim),
        )

    def forward(self, x):
        """
        NNの順方向の処理を行なう関数。
        モデルを呼び出したときに自動で呼び出されるため、直接呼び出してはいけない。

        Args:
            x: (バッチ数, 2) のテンソル

        Returns:
            各行動に対するQ値 (バッチ数, num_actions)
        """
        return self.model(x)
