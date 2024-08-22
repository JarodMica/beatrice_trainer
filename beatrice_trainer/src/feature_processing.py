
import os
import warnings
from typing import BinaryIO, Literal, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from beatrice_trainer.src.network_utils import dump_layer, ConvNeXtStack

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