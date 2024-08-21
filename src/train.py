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
import logging
import numpy as np
import pyworld
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.utils import *
from src.dataset import WavDataset, get_resampler
from src.feature_processing import PhoneExtractor, PitchEstimator
from src.vocoder import ConverterNetwork
from src.discriminator import MultiPeriodDiscriminator
from src.network_utils import dump_layer

def prepare_training(data_dir, out_dir, resume=False, config=None):
    # 各種準備をする
    # 副作用として、出力ディレクトリと TensorBoard のログファイルなどが生成される

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # (h, in_wav_dataset_dir, out_dir, resume) = (
    #     prepare_training_configs_for_experiment
    #     if is_notebook()
    #     else prepare_training_configs
    # )()
    
    (h, in_wav_dataset_dir, out_dir, resume) = prepare_training_configs(data_dir, out_dir, resume=False, config=None)

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
        # if not is_notebook():
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

def run_training(data_dir, out_dir, resume=False, config=None):
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
    ) = prepare_training(data_dir, out_dir, resume, config)
    
    global PARAPHERNALIA_VERSION
    
    # 学習
    for iteration in tqdm(range(initial_iteration, h.n_steps)):
        # === 1. データ前処理 ===
        try:
            batch = next(data_iter)
        except Exception as e:
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
        print(f"Starting iteration: {iteration}")
        if (iteration + 1) % 50000 == 0 or iteration + 1 in {
            1,
            5000,
            10000,
            30000,
            h.n_steps,
        }:
            print(f"Saving checkpoint at iteration: {iteration + 1}")
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



if __name__ == "__main__":
    data_dir = Path("data/model_1")
    out_dir = Path("trained_models/test1")
    run_training(data_dir, out_dir, resume=False, config=None)
    print("Training finished.")
    