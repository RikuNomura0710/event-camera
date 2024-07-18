# DL 基礎講座 2024 　最終課題「Optical Flow Prediction from Event Camera (EventCamera)」の提出用 github です。

## 環境構築

### requirements

- python 3.11

## ベースラインモデルを動かす

### 訓練・提出ファイル作成

```bash
python3 main.py
```

- `main.py`と同様のディレクトリ内に，学習したモデルのパラメータ`model.pth`とテストデータに対する予測結果`submission.npy`ファイルが作成されます．

### 各ファイルの説明

#### `main.py`

- **目的**: アプリケーションのエントリーポイント.モデルの初期化、トレーニングの実行、テスト、結果の保存を行う.
- **主要な関数**:
  - `set_seed`: シード値を設定する.
  - `compute_epe_error`: 予測されたオプティカルフローと正解データの end point error を計算する.
  - `save_optical_flow_to_npy`: オプティカルフローデータを `.npy` ファイルに保存する.
  - `main`: トレーニング, テスト, 予測の保存を行う.

#### `src/models/base.py`

- **目的**: ニューラルネットワークモデルの基本的な構成要素を含む.
- **主要なコンポーネント**:
  - `build_resnet_block`: ResNet スタイルのブロックを構築する.
  - `general_conv2d`: 汎用的な畳み込み層.
  - `upsample_conv2d_and_predict_flow`: 特徴マップをアップサンプルし、オプティカルフローを予測する.

#### `src/models/evflownet.py`

- **目的**: イベントベースのデータからオプティカルフローを予測するための `EVFlowNet` モデルの定義を含む.
- **主要なコンポーネント**:
  - `EVFlowNet`: オプティカルフローを予測するためのニューラルネットワークモデル.
  - `forward`: モデルのフォワードパスを定義する.

#### `src/datasets.py`

- **目的**: トレーニングとテスト用のデータセットをロードおよび前処理するためのユーティリティを提供する.
- **主要な関数**:
  - `train_collate`: トレーニング用のデータバッチを準備する.
  - `DataLoader`: データをロードする.
  - `Sequense`: シーケンスデータを格納する.

## DSEC Dataset (A Stereo Event Camera Dataset for Driving Scenarios) [[link](https://dsec.ifi.uzh.ch/)] の詳細

- 訓練データは合計 2015 データあり, イベントデータ，タイムスタンプ，正解オプティカルフローの RGB 画像が与えられる．
- 97 データがテストデータとして与えられる．
  - テストデータに対する回答は正解のオプティカルフローとし，訓練時には与えられない．

### データセットのダウンロード

- [こちら](https://drive.google.com/drive/folders/1xFVpggqbBxuwwy1MpIESyhhiKN052FJ1?usp=drive_link)から`train.zip`と`test.zip`をダウンロードし，`data/`ディレクトリに展開してください．

## タスクの詳細

- 本コンペでは，与えられたイベントデータ，適切なオプティカルフローをモデルに出力させる．
- 評価は以下の式で計算される EPE（End point Error）を用いる．
  $$\text{EPP}(\hat{u}, \hat{v}, u, v) = \sqrt{(\hat{u} - u)^2 + (\hat{v} - v)^2}\$$
  ここで：
- $\hat{u}\$ と $\hat{v}\$ は予測されたオプティカルフローの x 成分および y 成分.
- $u$ と $v$ は正解のオプティカルフローの x 成分および y 成分.

## 考えられる工夫の例

- 異なるスケールでのロスを足し合わせる．
  - ベースラインモデルは UNet 構造なので，デコーダーの中間層の出力は最終的な出力サイズの 0.5,0.25,...倍になっています．各中間層の出力を用いてロスを計算することで，勾配消失を防ぎ，性能向上が見込めます．
- 連続する 2 フレームを入力に用いる．
  - オプティカルフローはフレーム間でのピクセルの移動を表すベクトルです．したがって，入力に 2 フレーム（あるいはそれ以上）用いることで，イベントデータの変化量を用いてオプティカルフローの予測が可能になります．
- 画像の前処理．
  - 画像の前処理には形状を同じにするための Resize のみを利用しています．第 5 回の演習で紹介したようなデータ拡張を追加することで，疑似的にデータを増やし汎化性能の向上が見込めます．ただし，イベントデータは非常に疎なデータなので少し工夫が必要かもしれません．
