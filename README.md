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

# このリポジトリは現在非公開のはずです。見えている場合はProject Beatriceにご連絡ください

# Beatrice Trainer

超低遅延・低負荷・低容量を特徴とする完全無料の声質変換 VST 「Beatrice 2」の学習ツールキットです。

Beatrice 2 は、以下を目標に開発されています。

* 自分の変換された声を聴きながら、歌を快適に歌えるようにする
* 入力された声の抑揚を変換音声に正確に反映し、繊細な表現を可能にする
* 変換音声の高い自然性と明瞭さ
* 多様な変換先話者
* 公式 VST での変換時、外部の録音機器を使った実測で 50ms 程度の遅延
* 開発者のノート PC (Intel Core i7-1165G7) でシングルスレッドで動作させ、RTF < 0.25 となる程度の負荷
* 最小構成で 30MB 以下の容量
* VST と [VC Client](https://github.com/w-okada/voice-changer) での動作
* その他 (内緒)

## Getting Started

### 1. Download This Repo

Git などを使用して、このリポジトリをダウンロードしてください。

```sh
git lfs install
git clone https://huggingface.co/fierce-cats/beatrice-trainer
cd beatrice-trainer
```

### 2. Environment Setup

Poetry などを使用して、依存ライブラリをインストールしてください。

```sh
poetry install
poetry shell
# Alternatively, you can use pip to install dependencies directly:
# pip3 install -e .
```

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

学習の状況は、 TensorBoard で確認できます。

```sh
tensorboard --logdir <output_dir>
```

### 5. After Training

学習が正常に完了すると、出力ディレクトリ内に `paraphernalia_(data_dir_name)_(step)` という名前のディレクトリが生成されています。
このディレクトリを公式 VST や [VC Client](https://github.com/w-okada/voice-changer) で読み込むことで、ストリーム (リアルタイム) 変換を行うことができます。

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
* `checkpoint_(data_dir_name)_(step)`
  * 学習を途中から再開するためのチェックポイントです。
  * checkpoint_latest.pt にリネームし、 `-r` オプションを付けて学習スクリプトを実行すると、そのステップ数から学習を再開できます。
* `checkpoint_latest.pt`
  * 最も新しい checkpoint_(data_dir_name)_(step) のコピーです。
* `config.json`
  * 学習に使用されたコンフィグです。
* `events.out.tfevents.*`
  * TensorBoard で表示される情報を含むデータです。

### Customize Paraphernalia

学習スクリプトによって生成された paraphernalia ディレクトリ内にある `beatrice_paraphernalia_*.toml` ファイルを編集することで、 VST や VC Client 上での表示を変更できます。

`model.version` は、生成されたモデルのフォーマットバージョンを表すため、変更しないでください。

各 `description` は、長すぎると全文が表示されない場合があります。
現在表示できていても、将来的な VST や VC Client の仕様変更により表示できなくなる可能性があるため、余裕を持った文字数・行数に収めてください。

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
* [EnCodec](https://arxiv.org/abs/2210.13438) ([Official implementation](https://github.com/facebookresearch/encodec), [MIT License](https://github.com/facebookresearch/encodec/blob/main/LICENSE))
* [HiFi-GAN](https://arxiv.org/abs/2010.05646) ([Official implementation](https://github.com/jik876/hifi-gan), [MIT License](https://github.com/jik876/hifi-gan/blob/master/LICENSE))
* [Vocos](https://arxiv.org/abs/2306.00814) ([Official implementation](https://github.com/gemelo-ai/vocos), [MIT License](https://github.com/gemelo-ai/vocos/blob/main/LICENSE))
* [BigVSAN](https://arxiv.org/abs/2309.02836) ([Official implementation](https://github.com/sony/bigvsan), [MIT License](https://github.com/sony/bigvsan/blob/main/LICENSE))
* [UnivNet](https://arxiv.org/abs/2106.07889) ([Unofficial implementation](https://github.com/maum-ai/univnet), [BSD 3-Clause License](https://github.com/maum-ai/univnet/blob/master/LICENSE))
* [Soft-VC](https://arxiv.org/abs/2111.02392)
* [StreamVC](https://arxiv.org/abs/2401.03078)
* [EVA-GAN](https://arxiv.org/abs/2402.00892)
* [Subramani et al., 2024](https://arxiv.org/abs/2309.14507)
* [Agrawal et al., 2024](https://arxiv.org/abs/2401.10460)

## License

このリポジトリ内のソースコードおよび学習済みモデルは MIT License のもとで公開されています。
詳しくは [LICENSE](https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/LICENSE) をご覧ください。
