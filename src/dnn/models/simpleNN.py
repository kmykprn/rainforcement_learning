import torch
from torch import nn


class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()

        # 1次元に変換
        self.flatten = nn.Flatten()

        # 入力: 712次元, 出力: 10次元で処理
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        NNの順方向の処理を行なう関数。
        モデルを呼び出したときに自動で呼び出されるため、直接呼び出してはいけない。

        Args:
            x:
                (バッチ数, 28, 28)

        Returns:
            logits:
                (バッチ数, 10)
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class SimpleCNN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_output_class=10):
        super(SimpleCNN, self).__init__()

        # 畳み込み処理部分
        self.feature_extractor_stack = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 1次元に変換する処理
        self.flatten = nn.Flatten()

        # flatten後の次元数を自動で推定
        flatten_dim = self._get_conv_output_shape(input_shape)

        # 全結合処理
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_output_class),
        )

    def _get_conv_output_shape(self, input_shape):
        """
        畳み込み処理を行なった後、1次元化した後の次元数を取得する関数
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.feature_extractor_stack(dummy_input)
            return output.numel()  # flattenしたときの要素数

    def forward(self, x):
        """
        NNの順方向の処理を行なう関数。
        モデルを呼び出したときに自動で呼び出されるため、直接呼び出してはいけない。

        Args:
            x:
                (バッチ数, 1, 28, 28)

        Returns:
            logits:
                (バッチ数, 10)
        """
        x = self.feature_extractor_stack(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits
