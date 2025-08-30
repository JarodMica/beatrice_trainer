---
license: mit
pipeline_tag: audio-to-audio
tags:
- audio
- speech
- voice-conversion
datasets:
- reazon-research/reazonspeech
- dns-challenge
- libritts-r
---

# Beatrice Trainer

超低遅延・低負荷・低容量を特徴とする完全無料の声質変換 VST 「[Beatrice 2](https://prj-beatrice.com)」のモデル学習用ツールキットです。

Beatrice 2 は、以下を目標に開発されています。

* 自分の変換された声を聴きながら、歌を快適に歌えるようにする
* 入力された声の抑揚を変換音声に正確に反映し、繊細な表現を可能にする
* 変換音声の高い自然性と明瞭さ
* 多様な変換先話者
* 公式 VST での変換時、外部の録音機器を使った実測で 50ms 程度の遅延
* 開発者のノート PC (Intel Core i7-1165G7) でシングルスレッドで動作させ、 RTF < 0.2 となる程度の負荷
* 最小構成で 30MB 以下の容量
* VST と [VCClient](https://github.com/w-okada/voice-changer) での動作
* その他 (内緒)

## Release Notes

* **2025-08-31**: Beatrice Trainer 2.0.0-rc.0 をリリースしました。
  * **[公式 VST](https://prj-beatrice.com)、 [VCClient](https://github.com/w-okada/voice-changer)、 [beatrice-client](https://github.com/aq2r/beatrice-client) を最新版にアップデートしてください。新しい Trainer で生成したモデルは、古いバージョンの公式 VST、 VCClient、 beatrice-client で動作しません。**
  * RTF の目標値を 0.25 から 0.2 に変更しました。
  * パッケージマネージャを Poetry から uv に変更しました。
  * PitchEstimator の学習データに VocalSet を追加しました。
  * PitchEstimator の出力値の上限を A5 付近から F6 付近に引き上げました。
  * PitchEstimator が有声/無声の予測を行わないように変更しました。
  * PitchEstimator のアーキテクチャで、活性化関数が欠落していた箇所を修正しました。
  * PhoneExtractor のアーキテクチャに self-attention の追加や GRU の削除などの変更を行い、処理効率が向上しました。
  * WaveGenerator のアーキテクチャに cross-attention によって話者性を注入する構造を追加し、話者類似性が向上しました。
  * PhoneExtractor の出力に対して学習時にノイズを加算することにより、生成音声の品質が向上しました。
  * PhoneExtractor の出力に対する [kNN-VC](https://arxiv.org/abs/2305.18975) に類似したベクトル量子化処理を追加し、話者類似性が向上しました。
  * Discriminator に入力する波形に微細なノイズを加算する処理を追加し、学習の安定性が向上しました。
  * GradientEqualizer は品質への寄与が確認できなかったため、削除しました。
  * Data augmentation の処理にフォルマントシフトを追加し、話者類似性が向上しました。
  * Aperiodicity loss の計算における半フレームのずれを修正しました。
  * Aperiodicity loss を音量が非常に小さい部分では 0 とし、学習の安定性が向上しました。
  * Loudness loss を追加し、生成音声の品質が向上しました。
  * 学習率のスケジューリングを cosine から exponential に変更し、学習の延長が行いやすくなりました。
  * チェックポイントファイルを圧縮して保存するように変更しました。
  * コンフィグファイルで設定可能な項目を追加しました。
  * 損失関数の値などによって品質が評価できると誤解されることを避けるため、TensorBoard への数値の記録をデフォルトで無効にしました。
  * ハイパーパラメータの調整や、その他いくつかの変更を行いました。
* **2024-10-20**: Beatrice Trainer 2.0.0-beta.2 をリリースしました。
  * **[公式 VST](https://prj-beatrice.com) や [VCClient](https://github.com/w-okada/voice-changer) を最新版にアップデートしてください。新しい Trainer で生成したモデルは、古いバージョンの公式 VST や VCClient で動作しません。**
  * [Scaled Weight Standardization](https://arxiv.org/abs/2101.08692) の導入により、学習の安定性が向上しました。
  * 無音に非常に近い音声に対する損失の計算結果が nan になる問題を修正し、学習の安定性が向上しました。
  * 周期信号の生成方法を変更し、事前学習モデルを用いない場合により少ない学習ステップ数で高品質な変換音声を生成できるようになりました。
  * [FIRNet](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2024_ohtani.pdf) に着想を得たポストフィルタ構造を導入し、変換音声の品質が向上しました。
  * [D4C](https://www.sciencedirect.com/science/article/pii/S0167639316300413) を損失関数に導入し、変換音声の品質が向上しました。
  * [Multi-scale mel loss](https://arxiv.org/abs/2306.06546) を導入しました。
  * 冗長な逆伝播の除去や `torch.backends.cudnn.benchmark` の部分的な無効化などにより、学習速度が向上しました。
  * 学習データにモノラルでない音声ファイルが含まれる場合にエラーが発生する問題を修正しました。
  * 音量計算の誤りを修正し、学習時と推論時の変換結果の不一致が解消されました。
  * PyTorch のバージョンの下限を修正しました。
  * Windows 環境で CPU 版の PyTorch がインストールされる問題を修正しました。
  * Windows 環境で DataLoader の動作が非常に遅くなる問題を修正しました。
  * その他いくつかの変更を行いました。
* **2024-07-27**: Beatrice Trainer 2.0.0-beta.0 をリリースしました。


## Prerequisites

Beatrice は、既存の学習済みモデルを用いて声質の変換を行うだけであれば GPU を必要としません。
しかし、新たなモデルの作成を効率良く行うためには GPU が必要です。

学習スクリプトを実行すると、デフォルト設定では 9GB 程度の VRAM を消費します。
GeForce RTX 4090 を使用した場合、 40 分程度で学習が完了します。

GPU を手元に用意できない場合でも、以下のリポジトリを使用して Google Colab 上で学習を行うことができます。

* [w-okada/beatrice-trainer-colab](https://github.com/w-okada/beatrice-trainer-colab)

## Getting Started

### 1. Download This Repo

Git などを使用して、このリポジトリをダウンロードしてください。

```sh
git lfs install
git clone https://huggingface.co/fierce-cats/beatrice-trainer
cd beatrice-trainer
```

### 2. Environment Setup

uv などを使用して、依存ライブラリをインストールしてください。

```sh
uv sync --extra cu128
. .venv/bin/activate
# Alternatively, you can use pip to install dependencies directly:
# pip3 install -e .[cu128]
```
Windows 環境では、 `. .venv/bin/activate` の代わりに `.venv\Scripts\activate` を実行してください。

正しくインストールできていれば、 `python3 beatrice_trainer -h` で以下のようなヘルプが表示されます。

```
usage: beatrice_trainer [-h] [-d DATA_DIR] [-o OUT_DIR] [-r] [-c CONFIG]

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        directory containing the training data
  -o OUT_DIR, --out_dir OUT_DIR
                        output directory
  -r, --resume          resume training
  -c CONFIG, --config CONFIG
                        path to the config file
```

### 3. Prepare Your Training Data

下図のように学習データを配置してください。

```
your_training_data_dir
+---alice
|   +---alices_wonderful_speech.wav
|   +---alices_excellent_speech.flac // FLAC, MP3, and some other formats are also okay.
|   `---...
+---bob
|   +---bobs_fantastic_speech.wav
|   +---bobs_speeches
|   |   `---bobs_awesome_speech.wav // Audio files in nested directory will also be used.
|   `---...
`---...
```

学習データ用ディレクトリの直下に各話者のディレクトリを作る必要があります。
各話者のディレクトリの中の構造や音声ファイルの名前は自由です。

学習を行うデータが 1 話者のみの場合も、話者のディレクトリを作る必要があることに注意してください。

```
your_training_data_dir_with_only_one_speaker
+---charlies_brilliant_speech.wav // Wrong.
`---...
```

```
your_training_data_dir_with_only_one_speaker
`---charlie
    +---charlies_brilliant_speech.wav // Correct!
    `---...
```

### 4. Train Your Model

学習データを配置したディレクトリと出力ディレクトリを指定して学習を開始します。

```sh
python3 beatrice_trainer -d <your_training_data_dir> -o <output_dir>
```

(Windowns の場合、 `beatrice_trainer` の代わりに `.\beatrice_trainer\__main__.py` を指定しないと正しく動作しないという報告があります。)

学習の状況は、 TensorBoard で確認できます。

```sh
tensorboard --logdir <output_dir>
```

### 5. After Training

学習が正常に完了すると、出力ディレクトリ内に `paraphernalia_(data_dir_name)_(step)` という名前のディレクトリが生成されています。
このディレクトリを[公式 VST](https://prj-beatrice.com)、 [VCClient](https://github.com/w-okada/voice-changer) または [beatrice-client](https://github.com/aq2r/beatrice-client) で読み込むことで、ストリーム (リアルタイム) 変換を行うことができます。
**読み込めない場合は公式 VST、 VCClient、 beatrice-client のバージョンが古い可能性がありますので、最新のバージョンにアップデートしてください。**

## Detailed Usage

### Training

使用するハイパーパラメータや事前学習済みモデルをデフォルトと異なるものにする場合は、デフォルト値の書かれたコンフィグファイルである `assets/default_config.json` を別の場所にコピーして値を編集し、 `-c` でファイルを指定します。
`assets/default_config.json` を直接編集すると壊れるので注意してください。

また、コンフィグファイルに `data_dir` キーと `out_dir` キーを追加し、学習データを配置したディレクトリと出力ディレクトリを絶対パスまたはリポジトリルートからの相対パスで記載することで、コマンドライン引数での指定を省略できます。

```sh
python3 beatrice_trainer -c <your_config.json>
```

何らかの理由で学習が中断された場合、出力ディレクトリに `checkpoint_latest.pt` が生成されていれば、その学習を行っていたコマンドに `-r` オプションを追加して実行することで、最後に保存されたチェックポイントから学習を再開できます。

```sh
python3 beatrice_trainer -d <your_training_data_dir> -o <output_dir> -r
```

### Output Files

学習スクリプトを実行すると、出力ディレクトリ内に以下のファイル・ディレクトリが生成されます。

* `paraphernalia_(data_dir_name)_(step)`
  * ストリーム変換に必要なファイルを全て含むディレクトリです。
  * 学習途中のものも出力される場合があり、必要なステップ数のもの以外は削除して問題ありません。
  * このディレクトリ以外の出力物はストリーム変換に使用されないため、不要であれば削除して問題ありません。
* `checkpoint_(data_dir_name)_(step).pt.gz`
  * 学習を途中から再開するためのチェックポイントです。
  * checkpoint_latest.pt.gz にリネームし、 `-r` オプションを付けて学習スクリプトを実行すると、そのステップ数から学習を再開できます。
* `checkpoint_latest.pt.gz`
  * 最も新しい checkpoint_(data_dir_name)_(step).pt.gz のコピーです。
* `config.json`
  * 学習に使用されたコンフィグです。
* `events.out.tfevents.*`
  * TensorBoard で表示される情報を含むデータです。

### Customize Paraphernalia

学習スクリプトによって生成された paraphernalia ディレクトリ内にある `beatrice_paraphernalia_*.toml` ファイルを編集することで、 VST、 VCClient、 beatrice-client 上での表示を変更できます。

`model.version` は、生成されたモデルのフォーマットバージョンを表すため、変更しないでください。

各 `description` は、長すぎると全文が表示されない場合があります。
現在表示できていても、将来的な VST、 VCClient または beatrice-client の仕様変更により表示できなくなる可能性があるため、余裕を持った文字数・行数に収めてください。

`portrait` に設定する画像は、 PNG 形式かつ正方形としてください。

## Distribution of Trained Models

このリポジトリを用いて生成したモデルの配布を歓迎します。

配布されたモデルは、 Project Beatrice およびその関係者の管理する SNS アカウントやウェブサイト上でご紹介させていただく場合があります。
その際、 `portrait` に設定された画像を掲載することがありますので、予めご承知おきください。

## Resource

このリポジトリには、学習などに使用する各種データが含まれています。
詳しくは [assets/README.md](https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/assets/README.md) をご覧ください。

## Reference

* [wav2vec 2.0](https://arxiv.org/abs/2006.11477) ([Official implementation](https://github.com/facebookresearch/fairseq), [MIT License](https://github.com/facebookresearch/fairseq/blob/main/LICENSE))
  * FeatureExtractor の実装に利用。
* [EnCodec](https://arxiv.org/abs/2210.13438) ([Official implementation](https://github.com/facebookresearch/encodec), [MIT License](https://github.com/facebookresearch/encodec/blob/main/LICENSE))
  * GradBalancer の実装に利用。
* [HiFi-GAN](https://arxiv.org/abs/2010.05646) ([Official implementation](https://github.com/jik876/hifi-gan), [MIT License](https://github.com/jik876/hifi-gan/blob/master/LICENSE))
  * DiscriminatorP の実装に利用。
* [Vocos](https://arxiv.org/abs/2306.00814) ([Official implementation](https://github.com/gemelo-ai/vocos), [MIT License](https://github.com/gemelo-ai/vocos/blob/main/LICENSE))
  * ConvNeXtBlock の実装に利用。
* [BigVSAN](https://arxiv.org/abs/2309.02836) ([Official implementation](https://github.com/sony/bigvsan), [MIT License](https://github.com/sony/bigvsan/blob/main/LICENSE))
  * SAN モジュールの実装に利用。
* [D4C](https://www.sciencedirect.com/science/article/pii/S0167639316300413) ([Unofficial implementation by tuanad121](https://github.com/tuanad121/Python-WORLD), [MIT License](https://github.com/tuanad121/Python-WORLD/blob/master/LICENSE.txt))
  * 損失関数の実装に利用。
* [UnivNet](https://arxiv.org/abs/2106.07889) ([Unofficial implementation by maum-ai](https://github.com/maum-ai/univnet), [BSD 3-Clause License](https://github.com/maum-ai/univnet/blob/master/LICENSE))
  * DiscriminatorR の実装に利用。
* [FragmentVC](https://arxiv.org/abs/2010.14150)
  * SSL モデルに由来する特徴量をクエリとした cross-attention により声質を注入するアイデアを利用。
* [NF-ResNets](https://arxiv.org/abs/2101.08692)
  * Scaled Weight Standardization のアイデアを利用。
* [Soft-VC](https://arxiv.org/abs/2111.02392)
  * PhoneExtractor の基本的なアイデアとして利用。
* [kNN-VC](https://arxiv.org/abs/2305.18975)
  * 声質変換スキームを補助的にアイデアとして利用。
* [Descript Audio Codec](https://arxiv.org/abs/2306.06546)
  * Multi-scale mel loss のアイデアを利用。
* [StreamVC](https://arxiv.org/abs/2401.03078)
  * 声質変換スキームの基本的なアイデアとして利用。
* [FIRNet](https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2024_ohtani.pdf)
  * FIR フィルタを vocoder に適用するアイデアを利用。
* [EVA-GAN](https://arxiv.org/abs/2402.00892)
  * SiLU を vocoder に適用するアイデアを利用。
* [Subramani et al., 2024](https://arxiv.org/abs/2309.14507)
  * PitchEstimator の基本的なアイデアとして利用。
* [Agrawal et al., 2024](https://arxiv.org/abs/2401.10460)
  * Vocoder の基本的なアイデアとして利用。

## License

このリポジトリ内のソースコードおよび学習済みモデルは MIT License のもとで公開されています。
詳しくは [LICENSE](https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/LICENSE) をご覧ください。
