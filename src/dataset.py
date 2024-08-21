import os
from pathlib import Path
from typing import BinaryIO, Literal, Optional, Union

import math
from fractions import Fraction

import torch
import torchaudio
from torch.nn import functional as F

import logging

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
    

    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            assert voiced.numel() != 0, "Voiced tensor is empty"
            
            # 発声部分が中央にくるように、スライス区間を選ぶ
            center = voiced[torch.randint(0, voiced.numel(), ()).item()].item()
            slice_start = center - self.segment_length * self.out_hop_length // 2
            assert slice_start >= 0, f"Slice start is negative: {slice_start}"

            # スライス区間が含まれるように、ランダムに wav_length の長さを切り出す
            r = torch.randint(0, length - self.segment_length + 1, ()).item()
            offset = slice_start - r * self.out_hop_length

            clean_slice = clean_wav[offset : offset + self.wav_length]
            assert clean_slice.size(0) == self.wav_length, f"Clean wav slice size mismatch: {clean_slice.size(0)} != {self.wav_length}"
            clean_wavs.append(clean_slice)

            offset_in_sample_rate = int(round(offset * self.in_sample_rate / self.out_sample_rate))
            noisy_slice = noisy_wav[offset_in_sample_rate : offset_in_sample_rate + length * self.in_hop_length]
            assert noisy_slice.size(0) == length * self.in_hop_length, f"Noisy wav slice size mismatch: {noisy_slice.size(0)} != {length * self.in_hop_length}"
            noisy_wavs.append(noisy_slice)

            slice_starts.append(r)
            speaker_ids.append(speaker_id)
            formant_shifts.append(formant_shift)

        clean_wavs = torch.stack(clean_wavs)
        noisy_wavs = torch.stack(noisy_wavs)
        slice_starts = torch.tensor(slice_starts)
        speaker_ids = torch.tensor(speaker_ids)
        formant_shifts = torch.tensor(formant_shifts)

        logging.debug(f"Collate function output shapes: clean_wavs={clean_wavs.shape}, noisy_wavs={noisy_wavs.shape}, slice_starts={slice_starts.shape}, speaker_ids={speaker_ids.shape}, formant_shifts={formant_shifts.shape}")

        return (
            clean_wavs,  # [batch_size, wav_length]
            noisy_wavs,  # [batch_size, wav_length]
            slice_starts,  # Long[batch_size]
            speaker_ids,  # Long[batch_size]
            formant_shifts,  # Long[batch_size]
        )