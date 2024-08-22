import os
import warnings
import math
from typing import BinaryIO, Literal, Optional, Union
from fractions import Fraction

import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from beatrice_trainer.src.network_utils import dump_layer, ConvNeXtStack
from beatrice_trainer.src.feature_processing import PhoneExtractor, PitchEstimator

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