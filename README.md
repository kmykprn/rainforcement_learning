## これは何？
強化学習（深層学習連携前）のpython実装。
極力外部ライブラリは使わず実装した。

## 動作環境
Python 3.10.0

## 使い方
### 事前作業
```
cd reinforcement_learning
poetry install
```

### コードの実施：
各ディレクトリに train.py が格納されており、以下のように実行すると学習が始まる。
```
cd reinforcement_learning
poetry run python 03_Qlearning/train.py
```

### 各ディレクトリの説明：
- 01_MDP: マルコフ決定過程の実装
- 02_MonteCarlo: モンテカルロ法の実装
- 03_Qlearning: Qlearningの実装
- 04_SARSA: SARSAの実装
- 05_ActorCritic: ActorCriticの実装
- config: 学習回数などのハイパーパラメータを格納
- core: 強化学習のコア処理（ポリシー、環境など）を格納
- utils: 上記以外の共通処理を格納
