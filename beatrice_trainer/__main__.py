# %% [markdown]
# ## Settings

# %%
import argparse
import gc
import json
import math
import os
import shutil
import warnings
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from functools import partial
from pathlib import Path
from pprint import pprint
from random import Random
from typing import BinaryIO, Literal, Optional, Union

import numpy as np
import pyworld
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

assert "soundfile" in torchaudio.list_audio_backends()


# モジュールのバージョンではない
PARAPHERNALIA_VERSION = "2.0.0-alpha.2"


def is_notebook() -> bool:
    return "get_ipython" in globals()


def repo_root() -> Path:
    d = Path.cwd() / "dummy" if is_notebook() else Path(__file__)
    assert d.is_absolute(), d
    for d in d.parents:
        if (d / ".git").is_dir():
            return d
    raise RuntimeError("Repository root is not found.")


# ハイパーパラメータ
# 学習データや出力ディレクトリなど、学習ごとに変わるようなものはここに含めない
dict_default_hparams = {
    # train
    "learning_rate": 1e-4,
    "min_learning_rate": 5e-6,
    "adam_betas": [0.8, 0.99],
    "adam_eps": 1e-6,
    "batch_size": 8,
    "grad_weight_mel": 1.0,  # grad_weight は比が同じなら同じ意味になるはず
    "grad_weight_adv": 1.0,
    "grad_weight_fm": 1.0,
    "grad_balancer_ema_decay": 0.995,
    "use_amp": True,
    "num_workers": 16,
    "n_steps": 20000,
    "warmup_steps": 10000,
    "in_sample_rate": 16000,  # 変更不可
    "out_sample_rate": 24000,  # 変更不可
    "wav_length": 4 * 24000,  # 4s
    "segment_length": 100,  # 1s
    # data
    "phone_extractor_file": "assets/pretrained/003b_checkpoint_03000000.pt",
    "pitch_estimator_file": "assets/pretrained/008_1_checkpoint_00300000.pt",
    "in_ir_wav_dir": "assets/ir",
    "in_noise_wav_dir": "assets/noise",
    "in_test_wav_dir": "assets/test",
    "pretrained_file": "assets/pretrained/040c_checkpoint_libritts_r_200_02300000.pt",  # None も可
    # model
    "hidden_channels": 256,  # ファインチューン時変更不可、変更した場合は推論側の対応必要
    "san": False,  # ファインチューン時変更不可
}

if __name__ == "__main__":
    # スクリプト内部のデフォルト設定と assets/default_config.json が同期されているか確認
    default_config_file = repo_root() / "assets/default_config.json"
    if default_config_file.is_file():
        with open(default_config_file, encoding="utf-8") as f:
            default_config: dict = json.load(f)
        for key, value in dict_default_hparams.items():
            if key not in default_config:
                warnings.warn(f"{key} not found in default_config.json.")
            else:
                if value != default_config[key]:
                    warnings.warn(
                        f"{key} differs between default_config.json ({default_config[key]}) and internal default hparams ({value})."
                    )
                del default_config[key]
        for key in default_config:
            warnings.warn(f"{key} found in default_config.json is unknown.")
    else:
        warnings.warn("dafualt_config.json not found.")


def prepare_training_configs_for_experiment() -> tuple[dict, Path, Path, bool]:
    import ipynbname
    from IPython import get_ipython

    h = deepcopy(dict_default_hparams)
    in_wav_dataset_dir = repo_root() / "../../data/processed/libritts_r_200"
    try:
        notebook_name = ipynbname.name()
    except FileNotFoundError:
        notebook_name = Path(get_ipython().user_ns["__vsc_ipynb_file__"]).name
    out_dir = repo_root() / "notebooks" / notebook_name.split(".")[0].split("_")[0]
    resume = False
    return h, in_wav_dataset_dir, out_dir, resume


def prepare_training_configs() -> tuple[dict, Path, Path, bool]:
    # data_dir, out_dir は config ファイルでもコマンドライン引数でも指定でき、
    # コマンドライン引数が優先される。
    # 各種ファイルパスを相対パスで指定した場合、config ファイルでは
    # リポジトリルートからの相対パスとなるが、コマンドライン引数では
    # カレントディレクトリからの相対パスとなる。

    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-d", "--data_dir", type=Path, help="directory containing the training data")
    parser.add_argument("-o", "--out_dir", type=Path, help="output directory")
    parser.add_argument("-r", "--resume", action="store_true", help="resume training")
    parser.add_argument("-c", "--config", type=Path, help="path to the config file")
    # fmt: on
    args = parser.parse_args()

    # config
    if args.config is None:
        h = deepcopy(dict_default_hparams)
    else:
        with open(args.config, encoding="utf-8") as f:
            h = json.load(f)
    for key in dict_default_hparams.keys():
        if key not in h:
            h[key] = dict_default_hparams[key]
            warnings.warn(
                f"{key} is not specified in the config file. Using the default value."
            )
    # data_dir
    if args.data_dir is not None:
        in_wav_dataset_dir = args.data_dir
    elif "data_dir" in h:
        in_wav_dataset_dir = repo_root() / Path(h["data_dir"])
        del h["data_dir"]
    else:
        raise ValueError(
            "data_dir must be specified. "
            "For example `python3 beatrice_trainer -d my_training_data_dir -o my_output_dir`."
        )
    # out_dir
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif "out_dir" in h:
        out_dir = repo_root() / Path(h["out_dir"])
        del h["out_dir"]
    else:
        raise ValueError(
            "out_dir must be specified. "
            "For example `python3 beatrice_trainer -d my_training_data_dir -o my_output_dir`."
        )
    for key in list(h.keys()):
        if key not in dict_default_hparams:
            warnings.warn(f"`{key}` specified in the config file will be ignored.")
            del h[key]
    # resume
    resume = args.resume
    return h, in_wav_dataset_dir, out_dir, resume


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# %% [markdown]
# ## Phone Extractor


# %%
def dump_params(params: torch.Tensor, f: BinaryIO):
    if params is None:
        return
    if params.dtype == torch.bfloat16:
        f.write(
            params.detach()
            .clone()
            .float()
            .view(torch.short)
            .numpy()
            .ravel()[1::2]
            .tobytes()
        )
    else:
        f.write(params.detach().numpy().ravel().tobytes())
    f.flush()


def dump_layer(layer: nn.Module, f: BinaryIO):
    dump = partial(dump_params, f=f)
    if hasattr(layer, "dump"):
        layer.dump(f)
    elif isinstance(layer, (nn.Linear, nn.Conv1d, nn.LayerNorm)):
        dump(layer.weight)
        dump(layer.bias)
    elif isinstance(layer, nn.ConvTranspose1d):
        dump(layer.weight.transpose(0, 1))
        dump(layer.bias)
    elif isinstance(layer, nn.GRU):
        dump(layer.weight_ih_l0)
        dump(layer.bias_ih_l0)
        dump(layer.weight_hh_l0)
        dump(layer.bias_hh_l0)
        for i in range(1, 99999):
            if not hasattr(layer, f"weight_ih_l{i}"):
                break
            dump(getattr(layer, f"weight_ih_l{i}"))
            dump(getattr(layer, f"bias_ih_l{i}"))
            dump(getattr(layer, f"weight_hh_l{i}"))
            dump(getattr(layer, f"bias_hh_l{i}"))
    elif isinstance(layer, nn.Embedding):
        dump(layer.weight)
    elif isinstance(layer, nn.ModuleList):
        for l in layer:
            dump_layer(l, f)
    else:
        assert False, layer


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        delay: int = 0,
    ):
        padding = (kernel_size - 1) * dilation - delay
        self.trim = (kernel_size - 1) * dilation - 2 * delay
        if self.trim < 0:
            raise ValueError
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = super().forward(input)
        if self.trim == 0:
            return result
        else:
            return result[:, :, : -self.trim]


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        intermediate_channels: int,
        layer_scale_init_value: float,
        kernel_size: int = 7,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.use_weight_norm = use_weight_norm
        self.dwconv = CausalConv1d(
            channels, channels, kernel_size=kernel_size, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, intermediate_channels)
        self.pwconv2 = nn.Linear(intermediate_channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale_init_value))
        if use_weight_norm:
            self.norm = nn.Identity()
            self.dwconv = weight_norm(self.dwconv)
            self.pwconv1 = weight_norm(self.pwconv1)
            self.pwconv2 = weight_norm(self.pwconv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.pwconv2(x)
        x *= self.gamma
        x = x.transpose(1, 2)
        x += identity
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.dwconv)
            remove_weight_norm(self.pwconv1)
            remove_weight_norm(self.pwconv2)

    def merge_weights(self):
        if not self.use_weight_norm:
            self.pwconv1.bias.data += (
                self.norm.bias.data[None, :] * self.pwconv1.weight.data
            ).sum(1)
            self.pwconv1.weight.data *= self.norm.weight.data[None, :]
            self.norm.bias.data[:] = 0.0
            self.norm.weight.data[:] = 1.0
        self.pwconv2.weight.data *= self.gamma.data[:, None]
        self.pwconv2.bias.data *= self.gamma.data
        self.gamma.data[:] = 1.0

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.dwconv, f)
        dump_layer(self.pwconv1, f)
        dump_layer(self.pwconv2, f)


class ConvNeXtStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        intermediate_channels: int,
        n_blocks: int,
        delay: int,
        embed_kernel_size: int,
        kernel_size: int,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        assert delay * 2 + 1 <= embed_kernel_size
        self.use_weight_norm = use_weight_norm
        self.embed = CausalConv1d(in_channels, channels, embed_kernel_size, delay=delay)
        self.norm = nn.LayerNorm(channels)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    channels=channels,
                    intermediate_channels=intermediate_channels,
                    layer_scale_init_value=1.0 / n_blocks,
                    kernel_size=kernel_size,
                    use_weight_norm=use_weight_norm,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(channels)
        if use_weight_norm:
            self.embed = weight_norm(self.embed)
            self.norm = nn.Identity()
            self.final_layer_norm = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.embed)
            for conv_block in self.convnext:
                conv_block.remove_weight_norm()

    def merge_weights(self):
        for conv_block in self.convnext:
            conv_block.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.embed, f)
        if not self.use_weight_norm:
            dump_layer(self.norm, f)
        dump_layer(self.convnext, f)
        if not self.use_weight_norm:
            dump_layer(self.final_layer_norm, f)


class FeatureExtractor(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        # fmt: off
        self.conv0 = weight_norm(nn.Conv1d(1, hidden_channels // 8, 10, 5, bias=False))
        self.conv1 = weight_norm(nn.Conv1d(hidden_channels // 8, hidden_channels // 4, 3, 2, bias=False))
        self.conv2 = weight_norm(nn.Conv1d(hidden_channels // 4, hidden_channels // 2, 3, 2, bias=False))
        self.conv3 = weight_norm(nn.Conv1d(hidden_channels // 2, hidden_channels, 3, 2, bias=False))
        self.conv4 = weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, 2, bias=False))
        self.conv5 = weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 2, 2, bias=False))
        # fmt: on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, wav_length]
        wav_length = x.size(2)
        if wav_length % 160 != 0:
            warnings.warn("wav_length % 160 != 0")
        x = F.pad(x, (40, 40))
        x = F.gelu(self.conv0(x), approximate="tanh")
        x = F.gelu(self.conv1(x), approximate="tanh")
        x = F.gelu(self.conv2(x), approximate="tanh")
        x = F.gelu(self.conv3(x), approximate="tanh")
        x = F.gelu(self.conv4(x), approximate="tanh")
        x = F.gelu(self.conv5(x), approximate="tanh")
        # [batch_size, hidden_channels, wav_length / 160]
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv0)
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)
        remove_weight_norm(self.conv3)
        remove_weight_norm(self.conv4)
        remove_weight_norm(self.conv5)

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.conv0, f)
        dump_layer(self.conv1, f)
        dump_layer(self.conv2, f)
        dump_layer(self.conv3, f)
        dump_layer(self.conv4, f)
        dump_layer(self.conv5, f)


class FeatureProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.projection = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, channels, length]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.projection(x)
        x = self.dropout(x)
        return x

    def merge_weights(self):
        self.projection.bias.data += (
            (self.norm.bias.data[None, :, None] * self.projection.weight.data)
            .sum(1)
            .squeeze(1)
        )
        self.projection.weight.data *= self.norm.weight.data[None, :, None]
        self.norm.bias.data[:] = 0.0
        self.norm.weight.data[:] = 1.0

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.projection, f)


class PhoneExtractor(nn.Module):
    def __init__(
        self,
        phone_channels: int = 256,
        hidden_channels: int = 256,
        backbone_embed_kernel_size: int = 7,
        kernel_size: int = 17,
        n_blocks: int = 8,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(hidden_channels)
        self.feature_projection = FeatureProjection(hidden_channels, hidden_channels)
        self.n_speaker_encoder_layers = 3
        self.speaker_encoder = nn.GRU(
            hidden_channels,
            hidden_channels,
            self.n_speaker_encoder_layers,
            batch_first=True,
        )
        for i in range(self.n_speaker_encoder_layers):
            for input_char in "ih":
                self.speaker_encoder = weight_norm(
                    self.speaker_encoder, f"weight_{input_char}h_l{i}"
                )
        self.backbone = ConvNeXtStack(
            in_channels=hidden_channels,
            channels=hidden_channels,
            intermediate_channels=hidden_channels * 3,
            n_blocks=n_blocks,
            delay=0,
            embed_kernel_size=backbone_embed_kernel_size,
            kernel_size=kernel_size,
        )
        self.head = weight_norm(nn.Conv1d(hidden_channels, phone_channels, 1))

    def forward(
        self, x: torch.Tensor, return_stats: bool = True
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        # x: [batch_size, 1, wav_length]

        stats = {}

        # [batch_size, 1, wav_length] -> [batch_size, feature_extractor_hidden_channels, length]
        x = self.feature_extractor(x)
        if return_stats:
            stats["feature_norm"] = x.detach().norm(dim=1).mean()
        # [batch_size, feature_extractor_hidden_channels, length] -> [batch_size, hidden_channels, length]
        x = self.feature_projection(x)
        # [batch_size, hidden_channels, length] -> [batch_size, length, hidden_channels]
        g, _ = self.speaker_encoder(x.transpose(1, 2))
        if self.training:
            batch_size, length, _ = g.size()
            shuffle_sizes_for_each_data = torch.randint(
                0, 50, (batch_size,), device=g.device
            )
            max_indices = torch.arange(length, device=g.device)[None, :, None]
            min_indices = (
                max_indices - shuffle_sizes_for_each_data[:, None, None]
            ).clamp_(min=0)
            with torch.cuda.amp.autocast(False):
                indices = (
                    torch.rand(g.size(), device=g.device)
                    * (max_indices - min_indices + 1)
                ).long() + min_indices
            assert indices.min() >= 0, indices.min()
            assert indices.max() < length, (indices.max(), length)
            g = g.gather(1, indices)

        # [batch_size, length, hidden_channels] -> [batch_size, hidden_channels, length]
        g = g.transpose(1, 2).contiguous()
        # [batch_size, hidden_channels, length]
        x = self.backbone(x + g)
        # [batch_size, hidden_channels, length] -> [batch_size, phone_channels, length]
        phone = self.head(F.gelu(x, approximate="tanh"))

        results = [phone]
        if return_stats:
            stats["code_norm"] = phone.detach().norm(dim=1).mean().item()
            results.append(stats)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    @torch.inference_mode()
    def units(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, wav_length]

        # [batch_size, 1, wav_length] -> [batch_size, phone_channels, length]
        phone = self.forward(x, return_stats=False)
        # [batch_size, phone_channels, length] -> [batch_size, length, phone_channels]
        phone = phone.transpose(1, 2)
        # [batch_size, length, phone_channels]
        return phone

    def remove_weight_norm(self):
        self.feature_extractor.remove_weight_norm()
        for i in range(self.n_speaker_encoder_layers):
            for input_char in "ih":
                remove_weight_norm(self.speaker_encoder, f"weight_{input_char}h_l{i}")
        remove_weight_norm(self.head)

    def merge_weights(self):
        self.feature_projection.merge_weights()
        self.backbone.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.feature_extractor, f)
        dump_layer(self.feature_projection, f)
        dump_layer(self.speaker_encoder, f)
        dump_layer(self.backbone, f)
        dump_layer(self.head, f)


# %% [markdown]
# ## Pitch Estimator


# %%
def extract_pitch_features(
    y: torch.Tensor,  # [..., wav_length]
    hop_length: int = 160,  # 10ms
    win_length: int = 560,  # 35ms
    max_corr_period: int = 256,  # 16ms, 62.5Hz (16000 / 256)
    corr_win_length: int = 304,  # 19ms
    instfreq_features_cutoff_bin: int = 64,  # 1828Hz (16000 * 64 / 560)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert max_corr_period + corr_win_length == win_length

    # パディングする
    padding_length = (win_length - hop_length) // 2
    y = F.pad(y, (padding_length, padding_length))

    # フレームにする
    # [..., win_length, n_frames]
    y_frames = y.unfold(-1, win_length, hop_length).transpose_(-2, -1)

    # 複素スペクトログラム
    # Complex[..., (win_length // 2 + 1), n_frames]
    spec: torch.Tensor = torch.fft.rfft(y_frames, n=win_length, dim=-2)

    # Complex[..., instfreq_features_cutoff_bin, n_frames]
    spec = spec[..., :instfreq_features_cutoff_bin, :]

    # 対数パワースペクトログラム
    log_power_spec = spec.abs().add_(1e-5).log10_()

    # 瞬時位相の時間差分
    # 時刻 0 の値は 0
    delta_spec = spec[..., :, 1:] * spec[..., :, :-1].conj()
    delta_spec /= delta_spec.abs().add_(1e-5)
    delta_spec = torch.cat(
        [torch.zeros_like(delta_spec[..., :, :1]), delta_spec], dim=-1
    )

    # [..., instfreq_features_cutoff_bin * 3, n_frames]
    instfreq_features = torch.cat(
        [log_power_spec, delta_spec.real, delta_spec.imag], dim=-2
    )

    # 自己相関
    # 余裕があったら LPC 残差にするのも試したい
    # 元々これに 2.0 / corr_win_length を掛けて使おうと思っていたが、
    # この値は振幅の 2 乗に比例していて、NN に入力するために良い感じに分散を
    # 標準化する方法が思いつかなかったのでやめた
    flipped_y_frames = y_frames.flip((-2,))
    a = torch.fft.rfft(flipped_y_frames, n=win_length, dim=-2)
    b = torch.fft.rfft(y_frames[..., -corr_win_length:, :], n=win_length, dim=-2)
    # [..., max_corr_period, n_frames]
    corr = torch.fft.irfft(a * b, n=win_length, dim=-2)[..., corr_win_length:, :]

    # エネルギー項
    energy = flipped_y_frames.square_().cumsum_(-2)
    energy0 = energy[..., corr_win_length - 1 : corr_win_length, :]
    energy = energy[..., corr_win_length:, :] - energy[..., :-corr_win_length, :]

    # Difference function
    corr_diff = (energy0 + energy).sub_(corr.mul_(2.0))
    assert corr_diff.min() >= -1e-3, corr_diff.min()
    corr_diff.clamp_(min=0.0)  # 計算誤差対策

    # 標準化
    corr_diff *= 2.0 / corr_win_length
    corr_diff.sqrt_()

    # 変換モデルへの入力用のエネルギー
    energy = (
        y_frames.mul_(
            torch.signal.windows.cosine(win_length, device=y.device)[..., None]
        )
        .square_()
        .sum(-2, keepdim=True)
    )

    energy.clamp_(min=1e-3).log10_()  # >= -3, 振幅 1 の正弦波なら大体 2.15
    energy *= 0.5  # >= -1.5, 振幅 1 の正弦波なら大体 1.07, 1 の差は振幅で 20dB の差

    return (
        instfreq_features,  # [..., instfreq_features_cutoff_bin * 3, n_frames]
        corr_diff,  # [..., max_corr_period, n_frames]
        energy,  # [..., 1, n_frames]
    )


class PitchEstimator(nn.Module):
    def __init__(
        self,
        input_instfreq_channels: int = 192,
        input_corr_channels: int = 256,
        pitch_channels: int = 384,
        channels: int = 192,
        intermediate_channels: int = 192 * 3,
        n_blocks: int = 6,
        delay: int = 1,  # 10ms, 特徴抽出と合わせると 22.5ms
        embed_kernel_size: int = 3,
        kernel_size: int = 33,
        bins_per_octave: int = 96,
    ):
        super().__init__()
        self.bins_per_octave = bins_per_octave

        self.instfreq_embed_0 = nn.Conv1d(input_instfreq_channels, channels, 1)
        self.instfreq_embed_1 = nn.Conv1d(channels, channels, 1)
        self.corr_embed_0 = nn.Conv1d(input_corr_channels, channels, 1)
        self.corr_embed_1 = nn.Conv1d(channels, channels, 1)
        self.backbone = ConvNeXtStack(
            channels,
            channels,
            intermediate_channels,
            n_blocks,
            delay,
            embed_kernel_size,
            kernel_size,
        )
        self.head = nn.Conv1d(channels, pitch_channels, 1)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # wav: [batch_size, 1, wav_length]

        # [batch_size, input_instfreq_channels, length],
        # [batch_size, input_corr_channels, length]
        with torch.cuda.amp.autocast(False):
            instfreq_features, corr_diff, energy = extract_pitch_features(
                wav.squeeze(1),
                hop_length=160,
                win_length=560,
                max_corr_period=256,
                corr_win_length=304,
                instfreq_features_cutoff_bin=64,
            )
        instfreq_features = F.gelu(
            self.instfreq_embed_0(instfreq_features), approximate="tanh"
        )
        instfreq_features = self.instfreq_embed_1(instfreq_features)
        corr_diff = F.gelu(self.corr_embed_0(corr_diff), approximate="tanh")
        corr_diff = self.corr_embed_1(corr_diff)
        # [batch_size, channels, length]
        x = instfreq_features + corr_diff  # ここ活性化関数忘れてる
        x = self.backbone(x)
        # [batch_size, pitch_channels, length]
        x = self.head(x)
        return x, energy

    def sample_pitch(
        self, pitch: torch.Tensor, band_width: int = 48, return_features: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # pitch: [batch_size, pitch_channels, length]
        # 返されるピッチの値には 0 は含まれない
        batch_size, pitch_channels, length = pitch.size()
        pitch = pitch.softmax(1)
        if return_features:
            unvoiced_proba = pitch[:, :1, :].clone()
        pitch[:, 0, :] = -100.0
        pitch = (
            pitch.transpose(1, 2)
            .contiguous()
            .view(batch_size * length, 1, pitch_channels)
        )
        band_pitch = F.conv1d(
            pitch,
            torch.ones((1, 1, 1), device=pitch.device).expand(1, 1, band_width),
        )
        # [batch_size * length, 1, pitch_channels - band_width + 1] -> Long[batch_size * length, 1]
        quantized_band_pitch = band_pitch.argmax(2)
        if return_features:
            # [batch_size * length, 1]
            band_proba = band_pitch.gather(2, quantized_band_pitch[:, :, None])
            # [batch_size * length, 1]
            half_pitch_band_proba = band_pitch.gather(
                2,
                (quantized_band_pitch - self.bins_per_octave).clamp_(min=1)[:, :, None],
            )
            half_pitch_band_proba[quantized_band_pitch <= self.bins_per_octave] = 0.0
            half_pitch_proba = (half_pitch_band_proba / (band_proba + 1e-6)).view(
                batch_size, 1, length
            )
            # [batch_size * length, 1]
            double_pitch_band_proba = band_pitch.gather(
                2,
                (quantized_band_pitch + self.bins_per_octave).clamp_(
                    max=pitch_channels - band_width
                )[:, :, None],
            )
            double_pitch_band_proba[
                quantized_band_pitch
                > pitch_channels - band_width - self.bins_per_octave
            ] = 0.0
            double_pitch_proba = (double_pitch_band_proba / (band_proba + 1e-6)).view(
                batch_size, 1, length
            )
        # Long[1, pitch_channels]
        mask = torch.arange(pitch_channels, device=pitch.device)[None, :]
        # bool[batch_size * length, pitch_channels]
        mask = (quantized_band_pitch <= mask) & (
            mask < quantized_band_pitch + band_width
        )
        # Long[batch_size, length]
        quantized_pitch = (pitch.squeeze(1) * mask).argmax(1).view(batch_size, length)

        if return_features:
            features = torch.cat(
                [unvoiced_proba, half_pitch_proba, double_pitch_proba], dim=1
            )
            # Long[batch_size, length], [batch_size, 3, length]
            return quantized_pitch, features
        else:
            return quantized_pitch

    def merge_weights(self):
        self.backbone.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.instfreq_embed_0, f)
        dump_layer(self.instfreq_embed_1, f)
        dump_layer(self.corr_embed_0, f)
        dump_layer(self.corr_embed_1, f)
        dump_layer(self.backbone, f)
        dump_layer(self.head, f)


# %% [markdown]
# ## Vocoder


# %%
def overlap_add(
    ir: torch.Tensor,
    pitch: torch.Tensor,
    hop_length: int = 240,
    delay: int = 0,
) -> torch.Tensor:
    # print("ir, pitch: ", ir.dtype, pitch.dtype)
    batch_size, ir_length, length = ir.size()
    assert pitch.size() == (batch_size, length * hop_length)
    assert 0 <= delay < ir_length, (delay, ir_length)
    # 位相は [0, 1) で表す
    normalized_freq = pitch / 24000.0
    # 初期位相をランダムに設定
    normalized_freq[:, 0] = torch.rand(batch_size, device=pitch.device)
    with torch.cuda.amp.autocast(enabled=False):
        phase = (normalized_freq.double().cumsum_(1) % 1.0).float()
    # 重ねる箇所を求める
    # [n_pitchmarks], [n_pitchmarks]
    indices0, indices1 = torch.nonzero(phase[:, :-1] > phase[:, 1:], as_tuple=True)
    # 重ねる箇所の小数部分 (位相の遅れ) を求める
    numer = 1.0 - phase[indices0, indices1]
    # [n_pitchmarks]
    fractional_part = numer / (numer + phase[indices0, indices1 + 1])
    # 重ねる値を求める
    # [n_pitchmarks, ir_length]
    values = ir[indices0, :, indices1 // hop_length]
    # 位相を遅らせる
    # values が時間領域と仮定
    # Complex[n_pitchmarks, ir_length / 2 + 1]
    values = torch.fft.rfft(values, n=ir_length, dim=1)
    # 位相遅れの量
    # [n_pitchmarks, ir_length / 2 + 1]
    delay_phase = (
        torch.arange(ir_length // 2 + 1, device=pitch.device, dtype=torch.float32)[
            None, :
        ]
        / -ir_length
        * fractional_part[:, None]
    )
    # Complex[n_pitchmarks, ir_length / 2 + 1]
    delay_phase = torch.polar(torch.ones_like(delay_phase), delay_phase * math.tau)
    # values *= delay_phase
    values = values * delay_phase
    # [n_pitchmarks, ir_length]
    values = torch.fft.irfft(values, n=ir_length, dim=1)

    # 加算する値をサンプル単位にばらす
    # [n_pitchmarks * ir_length]
    values = values.ravel()
    # Long[n_pitchmarks * ir_length]
    indices0 = indices0[:, None].expand(-1, ir_length).ravel()
    # Long[n_pitchmarks * ir_length]
    indices1 = (
        indices1[:, None] + torch.arange(ir_length, device=pitch.device)
    ).ravel()

    # overlap-add する
    overlap_added_signal = torch.zeros(
        (batch_size, length * hop_length + ir_length), device=pitch.device
    )
    # print("overlap_added_signal, values: ", overlap_added_signal.dtype, values.dtype)
    overlap_added_signal.index_put_((indices0, indices1), values, accumulate=True)
    overlap_added_signal = overlap_added_signal[:, delay : -ir_length + delay]

    # sinc 重ねたものと ir を畳み込んだ方が FFT の回数減らせた気がする
    return overlap_added_signal


def generate_noise(aperiodicity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # aperiodicity: [batch_size, hop_length, length]
    batch_size, hop_length, length = aperiodicity.size()
    excitation = torch.rand(
        batch_size, (length + 1) * hop_length, device=aperiodicity.device
    )
    excitation -= 0.5
    n_fft = 2 * hop_length
    # 矩形窓で分析
    # Complex[batch_size, hop_length + 1, length]
    noise = torch.stft(
        excitation,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.ones(n_fft, device=excitation.device),
        center=False,
        return_complex=True,
    )
    assert noise.size(2) == aperiodicity.size(2), (
        noise.size(),
        aperiodicity.size(),
    )
    noise[:, 0, :] = 0.0
    noise[:, 1:, :] *= aperiodicity
    # ハン窓で合成
    # torch.istft は最適合成窓が使われるので使えないことに注意
    # [batch_size, 2 * hop_length, length]
    noise = torch.fft.irfft(noise, n=2 * hop_length, dim=1)
    noise *= torch.hann_window(2 * hop_length, device=noise.device)[None, :, None]
    # [batch_size, (length + 1) * hop_length]
    noise = F.fold(
        noise,
        (1, (length + 1) * hop_length),
        (1, 2 * hop_length),
        stride=(1, hop_length),
    ).squeeze_((1, 2))
    noise = noise[:, hop_length // 2 : -hop_length // 2]
    excitation = excitation[:, hop_length // 2 : -hop_length // 2]
    return noise, excitation  # [batch_size, length * hop_length]


class GradientEqualizerFunction(torch.autograd.Function):
    """ノルムが小さいほど勾配が大きくなってしまうのを補正する"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, length]
        rms = x.square().mean(dim=2, keepdim=True).sqrt_()
        ctx.save_for_backward(rms)
        return x

    @staticmethod
    def backward(ctx, dx: torch.Tensor) -> torch.Tensor:
        # dx: [batch_size, 1, length]
        (rms,) = ctx.saved_tensors
        dx = dx * (math.sqrt(2.0) * rms + 0.1)
        return dx


class PseudoDDSPVocoder(nn.Module):
    def __init__(
        self,
        channels: int,
        hop_length: int = 240,
        n_pre_blocks: int = 4,
    ):
        super().__init__()
        self.hop_length = hop_length

        self.prenet = ConvNeXtStack(
            in_channels=channels,
            channels=channels,
            intermediate_channels=channels * 3,
            n_blocks=n_pre_blocks,
            delay=2,  # 20ms 遅延
            embed_kernel_size=7,
            kernel_size=33,
        )
        self.ir_generator = ConvNeXtStack(
            in_channels=channels,
            channels=channels,
            intermediate_channels=channels * 3,
            n_blocks=2,
            delay=0,
            embed_kernel_size=3,
            kernel_size=33,
            use_weight_norm=True,
        )
        self.ir_generator_post = weight_norm(nn.Conv1d(channels, 512, 1, bias=False))
        self.aperiodicity_generator = ConvNeXtStack(
            in_channels=channels,
            channels=channels,
            intermediate_channels=channels * 3,
            n_blocks=2,
            delay=0,
            embed_kernel_size=3,
            kernel_size=33,
            use_weight_norm=True,
        )
        self.aperiodicity_generator_post = weight_norm(
            nn.Conv1d(channels, hop_length, 1, bias=False)
        )

    def forward(
        self, x: torch.Tensor, pitch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: [batch_size, channels, length]
        # pitch: [batch_size, length]

        x = self.prenet(x)
        ir = self.ir_generator(x)
        ir = F.elu(ir, inplace=True)
        # [batch_size, 512, length]
        ir = self.ir_generator_post(ir)

        # 最近傍補間
        # [batch_size, length * hop_length]
        pitch = torch.repeat_interleave(pitch, self.hop_length, dim=1)

        # [batch_size, length * hop_length]
        periodic_signal = overlap_add(ir, pitch, self.hop_length, delay=120)

        aperiodicity = self.aperiodicity_generator(x)
        aperiodicity = F.elu(aperiodicity, inplace=True)
        # [batch_size, hop_length, length]
        aperiodicity = self.aperiodicity_generator_post(aperiodicity)
        # [batch_size, length * hop_length], [batch_size, length * hop_length]
        aperiodic_signal, noise_excitation = generate_noise(aperiodicity)

        # [batch_size, 1, length * hop_length]
        y_g_hat = (periodic_signal + aperiodic_signal)[:, None, :]

        y_g_hat = GradientEqualizerFunction.apply(y_g_hat)

        return y_g_hat, {
            "periodic_signal": periodic_signal.detach(),
            "aperiodic_signal": aperiodic_signal.detach(),
            "noise_excitation": noise_excitation.detach(),
        }

    def remove_weight_norm(self):
        self.prenet.remove_weight_norm()
        self.ir_generator.remove_weight_norm()
        remove_weight_norm(self.ir_generator_post)
        self.aperiodicity_generator.remove_weight_norm()
        remove_weight_norm(self.aperiodicity_generator_post)

    def merge_weights(self):
        self.prenet.merge_weights()
        self.ir_generator.merge_weights()
        self.aperiodicity_generator.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.prenet, f)
        dump_layer(self.ir_generator, f)
        dump_layer(self.ir_generator_post, f)
        dump_layer(self.aperiodicity_generator, f)
        dump_layer(self.aperiodicity_generator_post, f)


def slice_segments(
    x: torch.Tensor, start_indices: torch.Tensor, segment_length: int
) -> torch.Tensor:
    batch_size, channels, _ = x.size()
    # [batch_size, 1, segment_size]
    indices = start_indices[:, None, None] + torch.arange(
        segment_length, device=start_indices.device
    )
    # [batch_size, channels, segment_size]
    indices = indices.expand(batch_size, channels, segment_length)
    return x.gather(2, indices)


class ConverterNetwork(nn.Module):
    def __init__(
        self,
        phone_extractor: PhoneExtractor,
        pitch_estimator: PitchEstimator,
        n_speakers: int,
        hidden_channels: int,
    ):
        super().__init__()
        self.frozen_modules = {
            "phone_extractor": phone_extractor.eval().requires_grad_(False),
            "pitch_estimator": pitch_estimator.eval().requires_grad_(False),
        }
        self.embed_phone = nn.Conv1d(256, hidden_channels, 1)
        self.embed_quantized_pitch = nn.Embedding(384, hidden_channels)
        phase = (
            torch.arange(384, dtype=torch.float)[:, None]
            * (
                torch.arange(0, hidden_channels, 2, dtype=torch.float)
                * (-math.log(10000.0) / hidden_channels)
            ).exp_()
        )
        self.embed_quantized_pitch.weight.data[:, 0::2] = phase.sin()
        self.embed_quantized_pitch.weight.data[:, 1::2] = phase.cos_()
        self.embed_quantized_pitch.weight.requires_grad_(False)
        self.embed_pitch_features = nn.Conv1d(4, hidden_channels, 1)
        self.embed_speaker = nn.Embedding(n_speakers, hidden_channels)
        self.embed_formant_shift = nn.Embedding(9, hidden_channels)
        self.vocoder = PseudoDDSPVocoder(
            channels=hidden_channels,
            hop_length=240,
            n_pre_blocks=4,
        )
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=720,
            hop_length=128,
            n_mels=80,
            power=2,  # 不安定さの原因になっているかも
            norm="slaney",
            mel_scale="slaney",
        )

    def _get_resampler(
        self, orig_freq, new_freq, device, cache={}
    ) -> torchaudio.transforms.Resample:
        key = orig_freq, new_freq
        if key in cache:
            return cache[key]
        resampler = torchaudio.transforms.Resample(orig_freq, new_freq).to(device)
        cache[key] = resampler
        return resampler

    def forward(
        self,
        x: torch.Tensor,
        target_speaker_id: torch.Tensor,
        formant_shift_semitone: torch.Tensor,
        pitch_shift_semitone: Optional[torch.Tensor] = None,
        slice_start_indices: Optional[torch.Tensor] = None,
        slice_segment_length: Optional[int] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        # x: [batch_size, 1, wav_length]
        # target_speaker_id: Long[batch_size]
        # formant_shift_semitone: [batch_size]
        # pitch_shift_semitone: [batch_size]
        # slice_start_indices: [batch_size]

        batch_size, _, _ = x.size()

        with torch.inference_mode():
            phone_extractor: PhoneExtractor = self.frozen_modules["phone_extractor"]
            pitch_estimator: PitchEstimator = self.frozen_modules["pitch_estimator"]
            # [batch_size, 1, wav_length] -> [batch_size, phone_channels, length]
            phone = phone_extractor.units(x).transpose(1, 2)
            # [batch_size, 1, wav_length] -> [batch_size, pitch_channels, length], [batch_size, 1, length]
            pitch, energy = pitch_estimator(x)
            # augmentation
            if self.training:
                # [batch_size, pitch_channels - 1]
                weights = pitch.softmax(1)[:, 1:, :].mean(2)
                # [batch_size]
                mean_pitch = (
                    weights * torch.arange(1, 384, device=weights.device)
                ).sum(1) / weights.sum(1)
                mean_pitch = mean_pitch.round_().long()
                target_pitch = torch.randint_like(mean_pitch, 64, 257)
                shift = target_pitch - mean_pitch
                shift_ratio = (
                    2.0 ** (shift.float() / pitch_estimator.bins_per_octave)
                ).tolist()
                shift = []
                interval_length = 100  # 1s
                interval_zeros = torch.zeros(
                    (1, 1, interval_length * 160), device=x.device
                )
                concatenated_shifted_x = []
                offsets = [0]
                for i in range(batch_size):
                    shift_ratio_i = shift_ratio[i]
                    shift_ratio_fraction_i = Fraction.from_float(
                        shift_ratio_i
                    ).limit_denominator(30)
                    shift_numer_i = shift_ratio_fraction_i.numerator
                    shift_denom_i = shift_ratio_fraction_i.denominator
                    shift_ratio_i = shift_numer_i / shift_denom_i
                    shift_i = int(
                        round(
                            math.log2(shift_ratio_i) * pitch_estimator.bins_per_octave
                        )
                    )
                    shift.append(shift_i)
                    shift_ratio[i] = shift_ratio_i
                    # [1, 1, wav_length / shift_ratio]
                    with torch.cuda.amp.autocast(False):
                        shifted_x_i = self._get_resampler(
                            shift_numer_i, shift_denom_i, x.device
                        )(x[i])[None]
                    if shifted_x_i.size(2) % 160 != 0:
                        shifted_x_i = F.pad(
                            shifted_x_i,
                            (0, 160 - shifted_x_i.size(2) % 160),
                            mode="reflect",
                        )
                    assert shifted_x_i.size(2) % 160 == 0
                    offsets.append(
                        offsets[-1] + interval_length + shifted_x_i.size(2) // 160
                    )
                    concatenated_shifted_x.extend([interval_zeros, shifted_x_i])
                if offsets[-1] % 256 != 0:
                    # 長さが同じ方が何かのキャッシュが効いて早くなるようなので
                    # 適当に 256 の倍数になるようにパディングして長さのパターン数を減らす
                    concatenated_shifted_x.append(
                        torch.zeros(
                            (1, 1, (256 - offsets[-1] % 256) * 160), device=x.device
                        )
                    )
                # [batch_size, 1, sum(wav_length) + batch_size * 16000]
                concatenated_shifted_x = torch.cat(concatenated_shifted_x, dim=2)
                assert concatenated_shifted_x.size(2) % (256 * 160) == 0
                # [1, pitch_channels, length / shift_ratio], [1, 1, length / shift_ratio]
                concatenated_pitch, concatenated_energy = pitch_estimator(
                    concatenated_shifted_x
                )
                for i in range(batch_size):
                    shift_i = shift[i]
                    shift_ratio_i = shift_ratio[i]
                    left = offsets[i] + interval_length
                    right = offsets[i + 1]
                    pitch_i = concatenated_pitch[:, :, left:right]
                    energy_i = concatenated_energy[:, :, left:right]
                    pitch_i = F.interpolate(
                        pitch_i,
                        scale_factor=shift_ratio_i,
                        mode="linear",
                        align_corners=False,
                    )
                    energy_i = F.interpolate(
                        energy_i,
                        scale_factor=shift_ratio_i,
                        mode="linear",
                        align_corners=False,
                    )
                    assert pitch_i.size(2) == energy_i.size(2)
                    assert abs(pitch_i.size(2) - pitch.size(2)) <= 10
                    length = min(pitch_i.size(2), pitch.size(2))

                    if shift_i > 0:
                        pitch[i : i + 1, :1, :length] = pitch_i[:, :1, :length]
                        pitch[i : i + 1, 1:-shift_i, :length] = pitch_i[
                            :, 1 + shift_i :, :length
                        ]
                        pitch[i : i + 1, -shift_i:, :length] = -10.0
                    elif shift_i < 0:
                        pitch[i : i + 1, :1, :length] = pitch_i[:, :1, :length]
                        pitch[i : i + 1, 1 : 1 - shift_i, :length] = -10.0
                        pitch[i : i + 1, 1 - shift_i :, :length] = pitch_i[
                            :, 1:shift_i, :length
                        ]
                    energy[i : i + 1, :, :length] = energy_i[:, :, :length]

            # [batch_size, pitch_channels, length] -> Long[batch_size, length], [batch_size, 3, length]
            quantized_pitch, pitch_features = pitch_estimator.sample_pitch(
                pitch, return_features=True
            )
            if pitch_shift_semitone is not None:
                quantized_pitch = torch.where(
                    quantized_pitch == 0,
                    quantized_pitch,
                    (
                        quantized_pitch
                        + (
                            pitch_shift_semitone[:, None]
                            * (pitch_estimator.bins_per_octave / 12)
                        )
                        .round_()
                        .long()
                    ).clamp_(1, 383),
                )
            pitch = 55.0 * 2.0 ** (
                quantized_pitch.float() / pitch_estimator.bins_per_octave
            )
            # phone が 2.5ms 先読みしているのに対して、
            # energy は 12.5ms, pitch_features は 22.5ms 先読みしているので、
            # ずらして phone に合わせる
            energy = F.pad(energy[:, :, :-1], (1, 0), mode="reflect")
            quantized_pitch = F.pad(quantized_pitch[:, :-2], (2, 0), mode="reflect")
            pitch_features = F.pad(pitch_features[:, :, :-2], (2, 0), mode="reflect")
            # [batch_size, 1, length], [batch_size, 3, length] -> [batch_size, 4, length]
            pitch_features = torch.cat([energy, pitch_features], dim=1)
            formant_shift_indices = (
                ((formant_shift_semitone + 2.0) * 2.0).round_().long()
            )

        phone = phone.clone()
        quantized_pitch = quantized_pitch.clone()
        pitch_features = pitch_features.clone()
        formant_shift_indices = formant_shift_indices.clone()
        pitch = pitch.clone()

        # [batch_sise, hidden_channels, length]
        x = (
            self.embed_phone(phone)
            + self.embed_quantized_pitch(quantized_pitch).transpose(1, 2)
            + self.embed_pitch_features(pitch_features)
            + (
                self.embed_speaker(target_speaker_id)[:, :, None]
                + self.embed_formant_shift(formant_shift_indices)[:, :, None]
            )
        )
        if slice_start_indices is not None:
            assert slice_segment_length is not None
            # [batch_size, hidden_channels, length] -> [batch_size, hidden_channels, segment_length]
            x = slice_segments(x, slice_start_indices, slice_segment_length)
        x = F.silu(x, inplace=True)
        # [batch_size, hidden_channels, segment_length] -> [batch_size, 1, segment_length * 240]
        y_g_hat, stats = self.vocoder(x, pitch)
        if return_stats:
            return y_g_hat, stats
        else:
            return y_g_hat

    def _normalize_melsp(self, x):
        return x.log().mul(0.5).clamp_(min=math.log(1e-5))

    def forward_and_compute_loss(
        self,
        noisy_wavs_16k: torch.Tensor,
        target_speaker_id: torch.Tensor,
        formant_shift_semitone: torch.Tensor,
        slice_start_indices: torch.Tensor,
        slice_segment_length: int,
        y_all: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # noisy_wavs_16k: [batch_size, 1, wav_length]
        # target_speaker_id: Long[batch_size]
        # formant_shift_semitone: [batch_size]
        # slice_start_indices: [batch_size]
        # slice_segment_length: int
        # y_all: [batch_size, 1, wav_length]

        # [batch_size, 1, wav_length] -> [batch_size, 1, wav_length * 240]
        y_hat_all, stats = self(
            noisy_wavs_16k,
            target_speaker_id,
            formant_shift_semitone,
            return_stats=True,
        )

        with torch.cuda.amp.autocast(False):
            melsp_periodic_signal = self.melspectrogram(
                stats["periodic_signal"].float()
            )
            melsp_aperiodic_signal = self.melspectrogram(
                stats["aperiodic_signal"].float()
            )
            melsp_noise_excitation = self.melspectrogram(
                stats["noise_excitation"].float()
            )
            # [1, n_mels, 1]
            # 1/6 ... [-0.5, 0.5] の一様乱数の平均パワー
            # 3/8 ... ハン窓をかけた時のパワー減衰
            # 0.5 ... 謎
            reference_melsp = self.melspectrogram.mel_scale(
                torch.full(
                    (1, self.melspectrogram.n_fft // 2 + 1, 1),
                    (1 / 6) * (3 / 8) * 0.5 * self.melspectrogram.win_length,
                    device=noisy_wavs_16k.device,
                )
            )
            aperiodic_ratio = melsp_aperiodic_signal / (
                melsp_periodic_signal + melsp_aperiodic_signal + 1e-5
            )
            compensation_ratio = reference_melsp / (melsp_noise_excitation + 1e-5)

            melsp_y_hat = self.melspectrogram(y_hat_all.float().squeeze(1))
            melsp_y_hat = melsp_y_hat * (
                (1.0 - aperiodic_ratio) + aperiodic_ratio * compensation_ratio
            )

            y_hat_mel = self._normalize_melsp(melsp_y_hat)
            # [batch_size, 1, wav_length] -> [batch_size, 1, wav_length * 240]
            y_hat = slice_segments(
                y_hat_all, slice_start_indices * 240, slice_segment_length * 240
            )

            y_mel = self._normalize_melsp(self.melspectrogram(y_all.squeeze(1)))
            # [batch_size, 1, wav_length] -> [batch_size, 1, wav_length * 240]
            y = slice_segments(
                y_all, slice_start_indices * 240, slice_segment_length * 240
            )

        loss_mel = F.l1_loss(y_hat_mel, y_mel)

        return y, y_hat, y_hat_all, loss_mel

    def remove_weight_norm(self):
        self.vocoder.remove_weight_norm()

    def merge_weights(self):
        self.vocoder.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.embed_phone, f)
        dump_layer(self.embed_quantized_pitch, f)
        dump_layer(self.embed_pitch_features, f)
        dump_layer(self.vocoder, f)


# Discriminator


def _normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min(1e-6)
    return tensor / denom


class SANConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        scale = self.weight.norm(p=2.0, dim=[1, 2, 3], keepdim=True).clamp_min(1e-6)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.zeros(in_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self, input: torch.Tensor, flg_san_train: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1, 1)
        if flg_san_train:
            out_fun = F.conv2d(
                input,
                normalized_weight.detach(),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out_dir = F.conv2d(
                input.detach(),
                normalized_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out = out_fun * scale, out_dir * scale.detach()
        else:
            out = F.conv2d(
                input,
                normalized_weight,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self) -> torch.Tensor:
        return _normalize(self.weight, dim=[1, 2, 3])


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class DiscriminatorP(nn.Module):
    def __init__(
        self, period: int, kernel_size: int = 5, stride: int = 3, san: bool = False
    ):
        super().__init__()
        self.period = period
        self.san = san
        # fmt: off
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (get_padding(kernel_size, 1), 0))),
        ])
        # fmt: on
        if san:
            self.conv_post = SANConv2d(1024, 1, (3, 1), 1, (1, 0))
        else:
            self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(
        self, x: torch.Tensor, flg_san_train: bool = False
    ) -> tuple[
        Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]
    ]:
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.silu(x, inplace=True)
            fmap.append(x)
        if self.san:
            x = self.conv_post(x, flg_san_train=flg_san_train)
        else:
            x = self.conv_post(x)
        if flg_san_train:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = x_fun, x_dir
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(nn.Module):
    def __init__(self, resolution: int, san: bool = False):
        super().__init__()
        self.resolution = resolution
        self.san = san
        assert len(self.resolution) == 3
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        if san:
            self.conv_post = SANConv2d(32, 1, (3, 3), padding=(1, 1))
        else:
            self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, flg_san_train: bool = False
    ) -> tuple[
        Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]
    ]:
        fmap = []

        x = self._spectrogram(x)
        x.unsqueeze_(1)
        for l in self.convs:
            x = l(x)
            x = F.silu(x, inplace=True)
            fmap.append(x)
        if self.san:
            x = self.conv_post(x, flg_san_train=flg_san_train)
        else:
            x = self.conv_post(x)
        if flg_san_train:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = x_fun, x_dir
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

        return x, fmap

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x, ((n_fft - hop_length) // 2, (n_fft - hop_length) // 2), mode="reflect"
        )
        x.squeeze_(1)
        with torch.cuda.amp.autocast(False):
            mag = torch.stft(
                x.float(),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.ones(win_length, device=x.device),
                center=False,
                return_complex=True,
            ).abs()

        return mag


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, san: bool = False):
        super().__init__()
        resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
        periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(r, san=san) for r in resolutions]
            + [DiscriminatorP(p, san=san) for p in periods]
        )
        self.discriminator_names = [f"R_{n}_{h}_{w}" for n, h, w in resolutions] + [
            f"P_{p}" for p in periods
        ]
        self.san = san

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, flg_san_train: bool = False
    ) -> tuple[
        list[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
        list[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        batch_size = y.size(0)
        concatenated_y_y_hat = torch.cat([y, y_hat])
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            if flg_san_train:
                (y_d_fun, y_d_dir), fmap = d(
                    concatenated_y_y_hat, flg_san_train=flg_san_train
                )
                y_d_r_fun, y_d_g_fun = torch.split(y_d_fun, batch_size)
                y_d_r_dir, y_d_g_dir = torch.split(y_d_dir, batch_size)
                y_d_r = y_d_r_fun, y_d_r_dir
                y_d_g = y_d_g_fun, y_d_g_dir
            else:
                y_d, fmap = d(concatenated_y_y_hat, flg_san_train=flg_san_train)
                y_d_r, y_d_g = torch.split(y_d, batch_size)
            fmap_r = []
            fmap_g = []
            for fm in fmap:
                fm_r, fm_g = torch.split(fm, batch_size)
                fmap_r.append(fm_r)
                fmap_g.append(fm_g)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def forward_and_compute_discriminator_loss(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        y_d_rs, y_d_gs, _, _ = self(y, y_hat, flg_san_train=self.san)
        loss = 0.0
        stats = {}
        assert len(y_d_gs) == len(y_d_rs) == len(self.discriminators)
        for dr, dg, name in zip(y_d_rs, y_d_gs, self.discriminator_names):
            if self.san:
                dr_fun, dr_dir = map(lambda x: x.float(), dr)
                dg_fun, dg_dir = map(lambda x: x.float(), dg)
                r_loss_fun = F.softplus(1.0 - dr_fun).square().mean()
                g_loss_fun = F.softplus(dg_fun).square().mean()
                r_loss_dir = F.softplus(1.0 - dr_dir).square().mean()
                g_loss_dir = -F.softplus(1.0 - dg_dir).square().mean()
                r_loss = r_loss_fun + r_loss_dir
                g_loss = g_loss_fun + g_loss_dir
            else:
                dr = dr.float()
                dg = dg.float()
                r_loss = (1.0 - dr).square().mean()
                g_loss = dg.square().mean()
            stats[f"{name}_dr_loss"] = r_loss.item()
            stats[f"{name}_dg_loss"] = g_loss.item()
            loss += r_loss + g_loss
        return loss, stats

    def forward_and_compute_generator_loss(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        _, y_d_gs, fmap_rs, fmap_gs = self(y, y_hat, flg_san_train=False)
        stats = {}
        # adversarial loss
        adv_loss = 0.0
        for dg, name in zip(y_d_gs, self.discriminator_names):
            dg = dg.float()
            if self.san:
                g_loss = F.softplus(1.0 - dg).square().mean()
            else:
                g_loss = (1.0 - dg).square().mean()
            stats[f"{name}_gg_loss"] = g_loss.item()
            adv_loss += g_loss
        # feature mathcing loss
        fm_loss = 0.0
        for fr, fg in zip(fmap_rs, fmap_gs):
            for r, g in zip(fr, fg):
                fm_loss += (r.detach() - g).abs().mean()
        return adv_loss, fm_loss, stats


# %% [markdown]
# ## Utilities


# %%
class GradBalancer:
    """Adapted from https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py"""

    def __init__(
        self,
        weights: dict[str, float],
        rescale_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
    ):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.ema_decay = ema_decay
        self.rescale_grads = rescale_grads

        self.ema_total: dict[str, float] = defaultdict(float)
        self.ema_fix: dict[str, float] = defaultdict(float)

    def backward(
        self,
        losses: dict[str, torch.Tensor],
        input: torch.Tensor,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        skip_update_ema: bool = False,
    ) -> dict[str, float]:
        stats = {}
        if skip_update_ema:
            assert len(losses) == len(self.ema_total)
            ema_norms = {k: tot / self.ema_fix[k] for k, tot in self.ema_total.items()}
        else:
            # 各 loss に対して d loss / d input とそのノルムを計算する
            norms = {}
            grads = {}
            for name, loss in losses.items():
                if scaler is not None:
                    loss = scaler.scale(loss)
                (grad,) = torch.autograd.grad(loss, [input], retain_graph=True)

                if not grad.isfinite().all():
                    input.backward(grad)
                    return {}
                grad = grad.detach() / (1.0 if scaler is None else scaler.get_scale())
                if self.per_batch_item:
                    dims = tuple(range(1, grad.dim()))
                    ema_norm = grad.norm(dim=dims).mean()
                else:
                    ema_norm = grad.norm()
                norms[name] = float(ema_norm)
                grads[name] = grad

            # ノルムの移動平均を計算する
            for key, value in norms.items():
                self.ema_total[key] = self.ema_total[key] * self.ema_decay + value
                self.ema_fix[key] = self.ema_fix[key] * self.ema_decay + 1.0
            ema_norms = {k: tot / self.ema_fix[k] for k, tot in self.ema_total.items()}

            # ログを取る
            total_ema_norm = sum(ema_norms.values())
            for k, ema_norm in ema_norms.items():
                stats[f"grad_norm_value_{k}"] = ema_norm
                stats[f"grad_norm_ratio_{k}"] = ema_norm / (total_ema_norm + 1e-12)

        # loss の係数の比率を計算する
        if self.rescale_grads:
            total_weights = sum([self.weights[k] for k in ema_norms])
            ratios = {k: w / total_weights for k, w in self.weights.items()}

        # 勾配を修正する
        loss = 0.0
        for name, ema_norm in ema_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (ema_norm + 1e-12)
            else:
                scale = self.weights[name]
            loss += (losses if skip_update_ema else grads)[name] * scale
        if scaler is not None:
            loss = scaler.scale(loss)
        if skip_update_ema:
            loss.backward()
        else:
            input.backward(loss)
        return stats

    def state_dict(self):
        return {
            "ema_total": self.ema_total,
            "ema_fix": self.ema_fix,
        }

    def load_state_dict(self, state_dict):
        self.ema_total = state_dict["ema_total"]
        self.ema_fix = state_dict["ema_fix"]


class QualityTester(nn.Module):
    def __init__(self):
        super().__init__()
        self.utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.0.0", "utmos22_strong", trust_repo=True
        ).eval()

    @torch.inference_mode()
    def compute_mos(self, wav: torch.Tensor) -> dict[str, list[float]]:
        res = {"utmos": self.utmos(wav, sr=16000).tolist()}
        return res

    def test(
        self, converted_wav: torch.Tensor, source_wav: torch.Tensor
    ) -> dict[str, list[float]]:
        # [batch_size, wav_length]
        res = {}
        res.update(self.compute_mos(converted_wav))
        return res

    def test_many(
        self, converted_wavs: list[torch.Tensor], source_wavs: list[torch.Tensor]
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        # list[batch_size, wav_length]
        results = defaultdict(list)
        assert len(converted_wavs) == len(source_wavs)
        for converted_wav, source_wav in zip(converted_wavs, source_wavs):
            res = self.test(converted_wav, source_wav)
            for metric_name, value in res.items():
                results[metric_name].extend(value)
        return {
            metric_name: sum(values) / len(values)
            for metric_name, values in results.items()
        }, results


def compute_grad_norm(
    model: nn.Module, return_stats: bool = False
) -> Union[float, dict[str, float]]:
    total_norm = 0.0
    stats = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm().item()
        if not math.isfinite(param_norm):
            param_norm = p.grad.data.float().norm().item()
        total_norm += param_norm * param_norm
        if return_stats:
            stats[f"grad_norm_{name}"] = param_norm
    total_norm = math.sqrt(total_norm)
    if return_stats:
        return total_norm, stats
    else:
        return total_norm


def compute_mean_f0(
    files: list[Path], method: Literal["dio", "harvest"] = "dio"
) -> float:
    sum_log_f0 = 0.0
    n_frames = 0
    for file in files:
        wav, sr = torchaudio.load(file, backend="soundfile")
        if method == "dio":
            f0, _ = pyworld.dio(wav.ravel().numpy().astype(np.float64), sr)
        elif method == "harvest":
            f0, _ = pyworld.harvest(wav.ravel().numpy().astype(np.float64), sr)
        else:
            raise ValueError(f"Invalid method: {method}")
        f0 = f0[f0 > 0]
        sum_log_f0 += float(np.log(f0).sum())
        n_frames += len(f0)
    if n_frames == 0:
        return math.nan
    mean_log_f0 = sum_log_f0 / n_frames
    return math.exp(mean_log_f0)


# %% [markdown]
# ## Dataset


# %%
def get_resampler(
    sr_before: int, sr_after: int, device="cpu", cache={}
) -> torchaudio.transforms.Resample:
    if not isinstance(device, str):
        device = str(device)
    if (sr_before, sr_after, device) not in cache:
        cache[(sr_before, sr_after, device)] = torchaudio.transforms.Resample(
            sr_before, sr_after
        ).to(device)
    return cache[(sr_before, sr_after, device)]


def convolve(signal: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    n = 1 << (signal.size(-1) + ir.size(-1) - 2).bit_length()
    res = torch.fft.irfft(torch.fft.rfft(signal, n=n) * torch.fft.rfft(ir, n=n), n=n)
    return res[..., : signal.size(-1)]


def random_filter(audio: torch.Tensor) -> torch.Tensor:
    assert audio.ndim == 2
    ab = torch.rand(audio.size(0), 6) * 0.75 - 0.375
    a, b = ab[:, :3], ab[:, 3:]
    a[:, 0] = 1.0
    b[:, 0] = 1.0
    audio = torchaudio.functional.lfilter(audio, a, b, clamp=False)
    return audio


def get_noise(
    n_samples: int, sample_rate: float, files: list[Union[str, bytes, os.PathLike]]
) -> torch.Tensor:
    resample_augmentation_candidates = [0.9, 0.95, 1.0, 1.05, 1.1]
    wavs = []
    current_length = 0
    while current_length < n_samples:
        idx_files = torch.randint(0, len(files), ())
        file = files[idx_files]
        wav, sr = torchaudio.load(file, backend="soundfile")
        assert wav.size(0) == 1
        augmented_sample_rate = int(
            round(
                sample_rate
                * resample_augmentation_candidates[
                    torch.randint(0, len(resample_augmentation_candidates), ())
                ]
            )
        )
        resampler = get_resampler(sr, augmented_sample_rate)
        wav = resampler(wav)
        wav = random_filter(wav)
        wav *= 0.99 / (wav.abs().max() + 1e-5)
        wavs.append(wav)
        current_length += wav.size(1)
    start = torch.randint(0, current_length - n_samples + 1, ())
    wav = torch.cat(wavs, dim=1)[:, start : start + n_samples]
    assert wav.size() == (1, n_samples), wav.size()
    return wav


def get_butterworth_lpf(
    cutoff_freq: int, sample_rate: int, cache={}
) -> tuple[torch.Tensor, torch.Tensor]:
    if (cutoff_freq, sample_rate) not in cache:
        q = math.sqrt(0.5)
        omega = math.tau * cutoff_freq / sample_rate
        cos_omega = math.cos(omega)
        alpha = math.sin(omega) / (2.0 * q)
        b1 = (1.0 - cos_omega) / (1.0 + alpha)
        b0 = b1 * 0.5
        a1 = -2.0 * cos_omega / (1.0 + alpha)
        a2 = (1.0 - alpha) / (1.0 + alpha)
        cache[(cutoff_freq, sample_rate)] = torch.tensor([b0, b1, b0]), torch.tensor(
            [1.0, a1, a2]
        )
    return cache[(cutoff_freq, sample_rate)]


def augment_audio(
    clean: torch.Tensor,
    sample_rate: int,
    noise_files: list[Union[str, bytes, os.PathLike]],
    ir_files: list[Union[str, bytes, os.PathLike]],
) -> torch.Tensor:
    # [1, wav_length]
    assert clean.size(0) == 1
    n_samples = clean.size(1)

    snr_candidates = [-20, -25, -30, -35, -40, -45]

    original_clean_rms = clean.square().mean().sqrt_()

    # noise を取得して clean と concat する
    noise = get_noise(n_samples, sample_rate, noise_files)
    signals = torch.cat([clean, noise])

    # clean, noise に異なるランダムフィルタをかける
    signals = random_filter(signals)

    # clean, noise にリバーブをかける
    if torch.rand(()) < 0.5:
        ir_file = ir_files[torch.randint(0, len(ir_files), ())]
        ir, sr = torchaudio.load(ir_file, backend="soundfile")
        assert ir.size() == (2, sr), ir.size()
        assert sr == sample_rate, (sr, sample_rate)
        signals = convolve(signals, ir)

    # clean, noise に同じ LPF をかける
    if torch.rand(()) < 0.2:
        if signals.abs().max() > 0.8:
            signals /= signals.abs().max() * 1.25
        cutoff_freq_candidates = [2000, 3000, 4000, 6000]
        cutoff_freq = cutoff_freq_candidates[
            torch.randint(0, len(cutoff_freq_candidates), ())
        ]
        b, a = get_butterworth_lpf(cutoff_freq, sample_rate)
        signals = torchaudio.functional.lfilter(signals, a, b, clamp=False)

    # clean の音量を合わせる
    clean, noise = signals
    clean_rms = clean.square().mean().sqrt_()
    clean *= original_clean_rms / clean_rms

    # clean, noise の音量をピークを重視して取る
    clean_level = clean.square().square_().mean().sqrt_().sqrt_()
    noise_level = noise.square().square_().mean().sqrt_().sqrt_()
    # SNR
    snr = snr_candidates[torch.randint(0, len(snr_candidates), ())]
    # noisy を生成
    noisy = clean + noise * (10.0 ** (snr / 20.0) * clean_level / (noise_level + 1e-5))
    return noisy


class WavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_files: list[tuple[Path, int]],
        in_sample_rate: int = 16000,
        out_sample_rate: int = 24000,
        wav_length: int = 4 * 24000,  # 4s
        segment_length: int = 100,  # 1s
        noise_files: Optional[list[Union[str, bytes, os.PathLike]]] = None,
        ir_files: Optional[list[Union[str, bytes, os.PathLike]]] = None,
    ):
        self.audio_files = audio_files
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.wav_length = wav_length
        self.segment_length = segment_length
        self.noise_files = noise_files
        self.ir_files = ir_files

        if (noise_files is None) is not (ir_files is None):
            raise ValueError("noise_files and ir_files must be both None or not None")

        self.in_hop_length = in_sample_rate // 100
        self.out_hop_length = out_sample_rate // 100  # 10ms 刻み

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        file, speaker_id = self.audio_files[index]
        clean_wav, sample_rate = torchaudio.load(file, backend="soundfile")

        formant_shift_candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        formant_shift = formant_shift_candidates[
            torch.randint(0, len(formant_shift_candidates), ()).item()
        ]

        resampler_fraction = Fraction(
            sample_rate / self.out_sample_rate * 2.0 ** (formant_shift / 12.0)
        ).limit_denominator(300)
        clean_wav = get_resampler(
            resampler_fraction.numerator, resampler_fraction.denominator
        )(clean_wav)

        assert clean_wav.size(0) == 1
        assert clean_wav.size(1) != 0

        clean_wav = F.pad(clean_wav, (self.wav_length, self.wav_length))

        if self.noise_files is None:
            assert False
            noisy_wav_16k = get_resampler(self.out_sample_rate, self.in_sample_rate)(
                clean_wav
            )
        else:
            clean_wav_16k = get_resampler(self.out_sample_rate, self.in_sample_rate)(
                clean_wav
            )
            noisy_wav_16k = augment_audio(
                clean_wav_16k, self.in_sample_rate, self.noise_files, self.ir_files
            )

        clean_wav = clean_wav.squeeze_(0)
        noisy_wav_16k = noisy_wav_16k.squeeze_(0)

        # 音量をランダマイズする
        amplitude = torch.rand(()).item() * 0.899 + 0.1
        factor = amplitude / clean_wav.abs().max()
        clean_wav *= factor
        noisy_wav_16k *= factor
        while noisy_wav_16k.abs().max() >= 1.0:
            clean_wav *= 0.5
            noisy_wav_16k *= 0.5

        return clean_wav, noisy_wav_16k, speaker_id, formant_shift

    def __len__(self) -> int:
        return len(self.audio_files)

    def collate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.wav_length % self.out_hop_length == 0
        length = self.wav_length // self.out_hop_length
        clean_wavs = []
        noisy_wavs = []
        slice_starts = []
        speaker_ids = []
        formant_shifts = []
        for clean_wav, noisy_wav, speaker_id, formant_shift in batch:
            # 発声部分をランダムに 1 箇所選ぶ
            (voiced,) = clean_wav.nonzero(as_tuple=True)
            assert voiced.numel() != 0
            center = voiced[torch.randint(0, voiced.numel(), ()).item()].item()
            # 発声部分が中央にくるように、スライス区間を選ぶ
            slice_start = center - self.segment_length * self.out_hop_length // 2
            assert slice_start >= 0
            # スライス区間が含まれるように、ランダムに wav_length の長さを切り出す
            r = torch.randint(0, length - self.segment_length + 1, ()).item()
            offset = slice_start - r * self.out_hop_length
            clean_wavs.append(clean_wav[offset : offset + self.wav_length])
            offset_in_sample_rate = int(
                round(offset * self.in_sample_rate / self.out_sample_rate)
            )
            noisy_wavs.append(
                noisy_wav[
                    offset_in_sample_rate : offset_in_sample_rate
                    + length * self.in_hop_length
                ]
            )
            slice_start = r
            slice_starts.append(slice_start)
            speaker_ids.append(speaker_id)
            formant_shifts.append(formant_shift)
        clean_wavs = torch.stack(clean_wavs)
        noisy_wavs = torch.stack(noisy_wavs)
        slice_starts = torch.tensor(slice_starts)
        speaker_ids = torch.tensor(speaker_ids)
        formant_shifts = torch.tensor(formant_shifts)
        return (
            clean_wavs,  # [batch_size, wav_length]
            noisy_wavs,  # [batch_size, wav_length]
            slice_starts,  # Long[batch_size]
            speaker_ids,  # Long[batch_size]
            formant_shifts,  # Long[batch_size]
        )


# %% [markdown]
# ## Train

# %%
AUDIO_FILE_SUFFIXES = {
    ".wav",
    ".aif",
    ".aiff",
    ".fla",
    ".flac",
    ".oga",
    ".ogg",
    ".opus",
    ".mp3",
}


def prepare_training():
    # 各種準備をする
    # 副作用として、出力ディレクトリと TensorBoard のログファイルなどが生成される

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    (h, in_wav_dataset_dir, out_dir, resume) = (
        prepare_training_configs_for_experiment
        if is_notebook()
        else prepare_training_configs
    )()

    print("config:")
    pprint(h)
    print()
    h = AttrDict(h)

    if not in_wav_dataset_dir.is_dir():
        raise ValueError(f"{in_wav_dataset_dir} is not found.")
    if resume:
        latest_checkpoint_file = out_dir / "checkpoint_latest.pt"
        if not latest_checkpoint_file.is_file():
            raise ValueError(f"{latest_checkpoint_file} is not found.")
    else:
        if out_dir.is_dir():
            if (out_dir / "checkpoint_latest.pt").is_file():
                raise ValueError(
                    f"{out_dir / 'checkpoint_latest.pt'} already exists. "
                    "Please specify a different output directory, or use --resume option."
                )
            for file in out_dir.iterdir():
                if file.suffix == ".pt":
                    raise ValueError(
                        f"{out_dir} already contains model files. "
                        "Please specify a different output directory."
                    )
        else:
            out_dir.mkdir(parents=True)

    in_ir_wav_dir = repo_root() / h.in_ir_wav_dir
    in_noise_wav_dir = repo_root() / h.in_noise_wav_dir
    in_test_wav_dir = repo_root() / h.in_test_wav_dir

    assert in_wav_dataset_dir.is_dir(), in_wav_dataset_dir
    assert out_dir.is_dir(), out_dir
    assert in_ir_wav_dir.is_dir(), in_ir_wav_dir
    assert in_noise_wav_dir.is_dir(), in_noise_wav_dir
    assert in_test_wav_dir.is_dir(), in_test_wav_dir

    # .wav または *.flac のファイルを再帰的に取得
    noise_files = sorted(
        list(in_noise_wav_dir.rglob("*.wav")) + list(in_noise_wav_dir.rglob("*.flac"))
    )
    if len(noise_files) == 0:
        raise ValueError(f"No audio data found in {in_noise_wav_dir}.")
    ir_files = sorted(
        list(in_ir_wav_dir.rglob("*.wav")) + list(in_ir_wav_dir.rglob("*.flac"))
    )
    if len(ir_files) == 0:
        raise ValueError(f"No audio data found in {in_ir_wav_dir}.")

    # TODO: 無音除去とか

    def get_training_filelist(in_wav_dataset_dir: Path):
        min_data_per_speaker = 1
        speakers: list[str] = []
        training_filelist: list[tuple[Path, int]] = []
        speaker_audio_files: list[list[Path]] = []
        for speaker_dir in sorted(in_wav_dataset_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            candidates = []
            for wav_file in sorted(speaker_dir.rglob("*")):
                if (
                    not wav_file.is_file()
                    or wav_file.suffix.lower() not in AUDIO_FILE_SUFFIXES
                ):
                    continue
                candidates.append(wav_file)
            if len(candidates) >= min_data_per_speaker:
                speaker_id = len(speakers)
                speakers.append(speaker_dir.name)
                training_filelist.extend([(file, speaker_id) for file in candidates])
                speaker_audio_files.append(candidates)
        return speakers, training_filelist, speaker_audio_files

    speakers, training_filelist, speaker_audio_files = get_training_filelist(
        in_wav_dataset_dir
    )
    n_speakers = len(speakers)
    if n_speakers == 0:
        raise ValueError(f"No speaker data found in {in_wav_dataset_dir}.")
    print(f"{n_speakers=}")
    for i, speaker in enumerate(speakers):
        print(f"  {i:{len(str(n_speakers - 1))}d}: {speaker}")
    print()
    print(f"{len(training_filelist)=}")

    def get_test_filelist(
        in_test_wav_dir: Path, n_speakers: int
    ) -> list[tuple[Path, list[int]]]:
        max_n_test_files = 1000
        test_filelist = []
        rng = Random(42)

        def get_target_id_generator():
            if n_speakers > 8:
                while True:
                    order = list(range(n_speakers))
                    rng.shuffle(order)
                    yield from order
            else:
                while True:
                    yield from range(n_speakers)

        target_id_generator = get_target_id_generator()
        for file in sorted(in_test_wav_dir.iterdir())[:max_n_test_files]:
            if file.suffix.lower() not in AUDIO_FILE_SUFFIXES:
                continue
            target_ids = [next(target_id_generator) for _ in range(min(8, n_speakers))]
            test_filelist.append((file, target_ids))
        return test_filelist

    test_filelist = get_test_filelist(in_test_wav_dir, n_speakers)
    if len(test_filelist) == 0:
        warnings.warn(f"No audio data found in {test_filelist}.")
    print(f"{len(test_filelist)=}")
    for file, target_ids in test_filelist[:12]:
        print(f"  {file}, {target_ids}")
    if len(test_filelist) > 12:
        print("  ...")
    print()

    # データ

    training_dataset = WavDataset(
        training_filelist,
        in_sample_rate=h.in_sample_rate,
        out_sample_rate=h.out_sample_rate,
        wav_length=h.wav_length,
        segment_length=h.segment_length,
        noise_files=noise_files,
        ir_files=ir_files,
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        num_workers=min(h.num_workers, os.cpu_count()),
        collate_fn=training_dataset.collate,
        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    print("Computing mean F0s of target speakers...", end="")
    speaker_f0s = []
    for speaker, files in enumerate(speaker_audio_files):
        if len(files) > 10:
            files = Random(42).sample(files, 10)
        f0 = compute_mean_f0(files)
        speaker_f0s.append(f0)
        if speaker % 5 == 0:
            print()
        print(f"  {speaker:3d}: {f0:.1f}Hz", end=",")
    print()
    print("Done.")
    print("Computing pitch shifts for test files...")
    test_pitch_shifts = []
    source_f0s = []
    for i, (file, target_ids) in enumerate(tqdm(test_filelist)):
        source_f0 = compute_mean_f0([file], method="harvest")
        source_f0s.append(source_f0)
        if source_f0 != source_f0:
            test_pitch_shifts.append([0] * len(target_ids))
            continue
        pitch_shifts = []
        for target_id in target_ids:
            target_f0 = speaker_f0s[target_id]
            if target_f0 != target_f0:
                pitch_shift = 0
            else:
                pitch_shift = int(round(12 * math.log2(target_f0 / source_f0)))
            pitch_shifts.append(pitch_shift)
        test_pitch_shifts.append(pitch_shifts)
    print("Done.")

    # モデルと最適化

    phone_extractor = PhoneExtractor().to(device).eval().requires_grad_(False)
    phone_extractor_checkpoint = torch.load(
        repo_root() / h.phone_extractor_file, map_location="cpu"
    )
    print(
        phone_extractor.load_state_dict(phone_extractor_checkpoint["phone_extractor"])
    )
    del phone_extractor_checkpoint

    pitch_estimator = PitchEstimator().to(device).eval().requires_grad_(False)
    pitch_estimator_checkpoint = torch.load(
        repo_root() / h.pitch_estimator_file, map_location="cpu"
    )
    print(
        pitch_estimator.load_state_dict(pitch_estimator_checkpoint["pitch_estimator"])
    )
    del pitch_estimator_checkpoint

    net_g = ConverterNetwork(
        phone_extractor,
        pitch_estimator,
        n_speakers,
        h.hidden_channels,
    ).to(device)
    net_d = MultiPeriodDiscriminator(san=h.san).to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        h.learning_rate,
        betas=h.adam_betas,
        eps=h.adam_eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        h.learning_rate,
        betas=h.adam_betas,
        eps=h.adam_eps,
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=h.use_amp)
    grad_balancer = GradBalancer(
        weights={
            "loss_mel": h.grad_weight_mel,
            "loss_adv": h.grad_weight_adv,
            "loss_fm": h.grad_weight_fm,
        },
        ema_decay=h.grad_balancer_ema_decay,
    )
    resample_to_in_sample_rate = torchaudio.transforms.Resample(
        h.out_sample_rate, h.in_sample_rate
    ).to(device)

    # チェックポイント読み出し

    initial_iteration = 0
    if resume:
        checkpoint_file = latest_checkpoint_file
    elif h.pretrained_file is not None:
        checkpoint_file = repo_root() / h.pretrained_file
    else:
        checkpoint_file = None
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        if not resume:  # ファインチューニング
            checkpoint_n_speakers = len(checkpoint["net_g"]["embed_speaker.weight"])
            initial_speaker_embedding = checkpoint["net_g"]["embed_speaker.weight"][:1]
            # initial_speaker_embedding = checkpoint["net_g"]["embed_speaker.weight"].mean(
            #     0, keepdim=True
            # )
            if True:
                # 0 とかランダムとかの方が良いかもしれない
                checkpoint["net_g"]["embed_speaker.weight"] = initial_speaker_embedding[
                    [0] * n_speakers
                ]
            else:  # 話者追加用
                assert n_speakers > checkpoint_n_speakers
                print(
                    f"embed_speaker.weight was padded: {checkpoint_n_speakers} -> {n_speakers}"
                )
                checkpoint["net_g"]["embed_speaker.weight"] = F.pad(
                    checkpoint["net_g"]["embed_speaker.weight"],
                    (0, 0, 0, n_speakers - checkpoint_n_speakers),
                )
                checkpoint["net_g"]["embed_speaker.weight"][
                    checkpoint_n_speakers:
                ] = initial_speaker_embedding
        print(net_g.load_state_dict(checkpoint["net_g"], strict=False))
        print(net_d.load_state_dict(checkpoint["net_d"], strict=False))
        if resume:
            optim_g.load_state_dict(checkpoint["optim_g"])
            optim_d.load_state_dict(checkpoint["optim_d"])
            initial_iteration = checkpoint["iteration"]
        grad_balancer.load_state_dict(checkpoint["grad_balancer"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    # スケジューラ

    def get_cosine_annealing_warmup_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_learning_rate: float,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        lr_ratio = min_learning_rate / optimizer.param_groups[0]["lr"]
        m = 0.5 * (1.0 - lr_ratio)
        a = 0.5 * (1.0 + lr_ratio)

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return current_epoch / warmup_epochs
            elif current_epoch < total_epochs:
                rate = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return math.cos(rate * math.pi) * m + a
            else:
                return min_learning_rate

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler_g = get_cosine_annealing_warmup_scheduler(
        optim_g, h.warmup_steps, h.n_steps, h.min_learning_rate
    )
    scheduler_d = get_cosine_annealing_warmup_scheduler(
        optim_d, h.warmup_steps, h.n_steps, h.min_learning_rate
    )
    for _ in range(initial_iteration + 1):
        scheduler_g.step()
        scheduler_d.step()

    net_g.train()
    net_d.train()

    # ログとか

    dict_scalars = defaultdict(list)
    quality_tester = QualityTester().eval().to(device)
    writer = SummaryWriter(out_dir)
    writer.add_text(
        "log",
        f"start training w/ {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'}.",
        initial_iteration,
    )
    if not resume:
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(dict(h), f, indent=4)
        if not is_notebook():
            shutil.copy(__file__, out_dir)

    return (
        device,
        in_wav_dataset_dir,
        h,
        out_dir,
        speakers,
        test_filelist,
        training_loader,
        speaker_f0s,
        test_pitch_shifts,
        phone_extractor,
        pitch_estimator,
        net_g,
        net_d,
        optim_g,
        optim_d,
        grad_scaler,
        grad_balancer,
        resample_to_in_sample_rate,
        initial_iteration,
        scheduler_g,
        scheduler_d,
        dict_scalars,
        quality_tester,
        writer,
    )


if __name__ == "__main__":
    (
        device,
        in_wav_dataset_dir,
        h,
        out_dir,
        speakers,
        test_filelist,
        training_loader,
        speaker_f0s,
        test_pitch_shifts,
        phone_extractor,
        pitch_estimator,
        net_g,
        net_d,
        optim_g,
        optim_d,
        grad_scaler,
        grad_balancer,
        resample_to_in_sample_rate,
        initial_iteration,
        scheduler_g,
        scheduler_d,
        dict_scalars,
        quality_tester,
        writer,
    ) = prepare_training()

    # 学習

    for iteration in tqdm(range(initial_iteration, h.n_steps)):
        # === 1. データ前処理 ===
        try:
            batch = next(data_iter)
        except:
            data_iter = iter(training_loader)
            batch = next(data_iter)
        (
            clean_wavs,
            noisy_wavs_16k,
            slice_starts,
            speaker_ids,
            formant_shift_semitone,
        ) = map(lambda x: x.to(device, non_blocking=True), batch)

        # === 2.1 Discriminator の学習 ===

        with torch.cuda.amp.autocast(h.use_amp):
            # Generator
            y, y_hat, y_hat_for_backward, loss_mel = net_g.forward_and_compute_loss(
                noisy_wavs_16k[:, None, :],
                speaker_ids,
                formant_shift_semitone,
                slice_start_indices=slice_starts,
                slice_segment_length=h.segment_length,
                y_all=clean_wavs[:, None, :],
            )
            assert y_hat.isfinite().all()
            assert loss_mel.isfinite().all()

            # Discriminator
            loss_discriminator, discriminator_d_stats = (
                net_d.forward_and_compute_discriminator_loss(y, y_hat.detach())
            )

        optim_d.zero_grad()
        grad_scaler.scale(loss_discriminator).backward()
        grad_scaler.unscale_(optim_d)
        grad_norm_d, d_grad_norm_stats = compute_grad_norm(net_d, True)
        grad_scaler.step(optim_d)

        # === 2.2 Generator の学習 ===

        with torch.cuda.amp.autocast(h.use_amp):
            # Discriminator
            loss_adv, loss_fm, discriminator_g_stats = (
                net_d.forward_and_compute_generator_loss(y, y_hat)
            )

        optim_g.zero_grad()
        gradient_balancer_stats = grad_balancer.backward(
            {
                "loss_mel": loss_mel,
                "loss_adv": loss_adv,
                "loss_fm": loss_fm,
            },
            y_hat_for_backward,
            grad_scaler,
            skip_update_ema=iteration > 10 and iteration % 5 != 0,
        )
        grad_scaler.unscale_(optim_g)
        grad_norm_g, g_grad_norm_stats = compute_grad_norm(net_g, True)
        grad_scaler.step(optim_g)
        grad_scaler.update()

        # === 3. ログ ===

        dict_scalars["loss_g/loss_mel"].append(loss_mel.item())
        dict_scalars["loss_g/loss_fm"].append(loss_fm.item())
        dict_scalars["loss_g/loss_adv"].append(loss_adv.item())
        dict_scalars["other/grad_scale"].append(grad_scaler.get_scale())
        dict_scalars["loss_d/loss_discriminator"].append(loss_discriminator.item())
        if math.isfinite(grad_norm_d):
            dict_scalars["other/gradient_norm_d"].append(grad_norm_d)
            for name, value in d_grad_norm_stats.items():
                dict_scalars[f"~gradient_norm_d/{name}"].append(value)
        if math.isfinite(grad_norm_g):
            dict_scalars["other/gradient_norm_g"].append(grad_norm_g)
            for name, value in g_grad_norm_stats.items():
                dict_scalars[f"~gradient_norm_g/{name}"].append(value)
        dict_scalars["other/lr_g"].append(scheduler_g.get_last_lr()[0])
        dict_scalars["other/lr_d"].append(scheduler_d.get_last_lr()[0])
        for k, v in discriminator_d_stats.items():
            dict_scalars[f"~loss_discriminator/{k}"].append(v)
        for k, v in discriminator_g_stats.items():
            dict_scalars[f"~loss_discriminator/{k}"].append(v)
        for k, v in gradient_balancer_stats.items():
            dict_scalars[f"~gradient_balancer/{k}"].append(v)

        if (iteration + 1) % 1000 == 0 or iteration == 0:
            for name, scalars in dict_scalars.items():
                if scalars:
                    writer.add_scalar(name, sum(scalars) / len(scalars), iteration + 1)
                    scalars.clear()

        # === 4. 検証 ===
        if (iteration + 1) % 50000 == 0 or iteration + 1 in {
            1,
            5000,
            10000,
            30000,
            h.n_steps,
        }:
            net_g.eval()
            torch.cuda.empty_cache()

            dict_qualities_all = defaultdict(list)
            n_added_wavs = 0
            with torch.inference_mode():
                for i, ((file, target_ids), pitch_shift_semitones) in enumerate(
                    zip(test_filelist, test_pitch_shifts)
                ):
                    source_wav, sr = torchaudio.load(file, backend="soundfile")
                    source_wav = source_wav.to(device)
                    if sr != h.in_sample_rate:
                        source_wav = get_resampler(sr, h.in_sample_rate, device)(
                            source_wav
                        )
                    source_wav = source_wav.to(device)
                    original_source_wav_length = source_wav.size(1)
                    # 長さのパターンを減らしてキャッシュを効かせる
                    if source_wav.size(1) % h.in_sample_rate == 0:
                        padded_source_wav = source_wav
                    else:
                        padded_source_wav = F.pad(
                            source_wav,
                            (
                                0,
                                h.in_sample_rate
                                - source_wav.size(1) % h.in_sample_rate,
                            ),
                        )
                    converted = net_g(
                        padded_source_wav[[0] * len(target_ids), None],
                        torch.tensor(target_ids, device=device),
                        torch.tensor(
                            [0.0] * len(target_ids), device=device
                        ),  # フォルマントシフト
                        torch.tensor(
                            [float(p) for p in pitch_shift_semitones], device=device
                        ),
                    ).squeeze_(1)[:, : original_source_wav_length // 160 * 240]
                    if i < 12:
                        if iteration == 0:
                            writer.add_audio(
                                f"source/y_{i:02d}",
                                source_wav,
                                iteration + 1,
                                h.in_sample_rate,
                            )
                        for d in range(
                            min(len(target_ids), 1 + (12 - i - 1) // len(test_filelist))
                        ):
                            idx_in_batch = n_added_wavs % len(target_ids)
                            writer.add_audio(
                                f"converted/y_hat_{i:02d}_{target_ids[idx_in_batch]:03d}_{pitch_shift_semitones[idx_in_batch]:+02d}",
                                converted[idx_in_batch],
                                iteration + 1,
                                h.out_sample_rate,
                            )
                            n_added_wavs += 1
                    converted = resample_to_in_sample_rate(converted)
                    quality = quality_tester.test(converted, source_wav)
                    for metric_name, values in quality.items():
                        dict_qualities_all[metric_name].extend(values)
            assert n_added_wavs == min(
                12, len(test_filelist) * len(test_filelist[0][1])
            ), (
                n_added_wavs,
                len(test_filelist),
                len(speakers),
                len(test_filelist[0][1]),
            )
            dict_qualities = {
                metric_name: sum(values) / len(values)
                for metric_name, values in dict_qualities_all.items()
                if len(values)
            }
            for metric_name, value in dict_qualities.items():
                writer.add_scalar(f"validation/{metric_name}", value, iteration + 1)
            for metric_name, values in dict_qualities_all.items():
                for i, value in enumerate(values):
                    writer.add_scalar(
                        f"~validation_{metric_name}/{i:03d}", value, iteration + 1
                    )
            del dict_qualities, dict_qualities_all

            gc.collect()
            net_g.train()
            torch.cuda.empty_cache()

        # === 5. 保存 ===
        if (iteration + 1) % 50000 == 0 or iteration + 1 in {
            1,
            5000,
            10000,
            30000,
            h.n_steps,
        }:
            # チェックポイント
            name = f"{in_wav_dataset_dir.name}_{iteration + 1:08d}"
            checkpoint_file_save = out_dir / f"checkpoint_{name}.pt"
            if checkpoint_file_save.exists():
                checkpoint_file_save = checkpoint_file_save.with_name(
                    f"{checkpoint_file_save.name}_{hash(None):x}"
                )
            torch.save(
                {
                    "iteration": iteration + 1,
                    "net_g": net_g.state_dict(),
                    "phone_extractor": phone_extractor.state_dict(),
                    "pitch_estimator": pitch_estimator.state_dict(),
                    "net_d": net_d.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "grad_balancer": grad_balancer.state_dict(),
                    "grad_scaler": grad_scaler.state_dict(),
                    "h": dict(h),
                },
                checkpoint_file_save,
            )
            shutil.copy(checkpoint_file_save, out_dir / "checkpoint_latest.pt")

            # 推論用
            paraphernalia_dir = out_dir / f"paraphernalia_{name}"
            if paraphernalia_dir.exists():
                paraphernalia_dir = paraphernalia_dir.with_name(
                    f"{paraphernalia_dir.name}_{hash(None):x}"
                )
            paraphernalia_dir.mkdir()
            phone_extractor_fp16 = PhoneExtractor()
            phone_extractor_fp16.load_state_dict(phone_extractor.state_dict())
            phone_extractor_fp16.remove_weight_norm()
            phone_extractor_fp16.merge_weights()
            phone_extractor_fp16.half()
            phone_extractor_fp16.dump(paraphernalia_dir / f"phone_extractor.bin")
            del phone_extractor_fp16
            pitch_estimator_fp16 = PitchEstimator()
            pitch_estimator_fp16.load_state_dict(pitch_estimator.state_dict())
            pitch_estimator_fp16.merge_weights()
            pitch_estimator_fp16.half()
            pitch_estimator_fp16.dump(paraphernalia_dir / f"pitch_estimator.bin")
            del pitch_estimator_fp16
            net_g_fp16 = ConverterNetwork(
                nn.Module(), nn.Module(), len(speakers), h.hidden_channels
            )
            net_g_fp16.load_state_dict(net_g.state_dict())
            net_g_fp16.remove_weight_norm()
            net_g_fp16.merge_weights()
            net_g_fp16.half()
            net_g_fp16.dump(paraphernalia_dir / f"waveform_generator.bin")
            with open(paraphernalia_dir / f"speaker_embeddings.bin", "wb") as f:
                dump_layer(net_g_fp16.embed_speaker, f)
            with open(paraphernalia_dir / f"formant_shift_embeddings.bin", "wb") as f:
                dump_layer(net_g_fp16.embed_formant_shift, f)
            del net_g_fp16
            shutil.copy(repo_root() / "assets/images/noimage.png", paraphernalia_dir)
            with open(
                paraphernalia_dir / f"beatrice_paraphernalia_{name}.toml",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    f'''[model]
version = "{PARAPHERNALIA_VERSION}"
name = "{name}"
description = """
No description for this model.
このモデルの説明はありません。
"""
'''
                )
                for speaker_id, (speaker, speaker_f0) in enumerate(
                    zip(speakers, speaker_f0s)
                ):
                    average_pitch = 69.0 + 12.0 * math.log2(speaker_f0 / 440.0)
                    average_pitch = round(average_pitch * 8.0) / 8.0
                    f.write(
                        f'''
[voice.{speaker_id}]
name = "{speaker}"
description = """
No description for this voice.
この声の説明はありません。
"""
average_pitch = {average_pitch}

[voice.{speaker_id}.portrait]
path = "noimage.png"
description = """
"""
'''
                    )
            del paraphernalia_dir

        # TODO: phone_extractor, pitch_estimator が既知のモデルであれば dump を省略

        # === 6. スケジューラ更新 ===
        scheduler_g.step()
        scheduler_d.step()


print("Training finished.")
