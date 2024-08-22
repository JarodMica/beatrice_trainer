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

from beatrice_trainer.src.dataset import WavDataset
from beatrice_trainer.src.feature_processing import PhoneExtractor, PitchEstimator

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

PARAPHERNALIA_VERSION = "2.0.0-alpha.2"

dict_default_hparams = {
    # train
    "learning_rate": 1e-4,
    "min_learning_rate": 5e-6,
    "adam_betas": [0.8, 0.99],
    "adam_eps": 1e-6,
    "batch_size": 32,
    "grad_weight_mel": 1.0,  # grad_weight は比が同じなら同じ意味になるはず
    "grad_weight_adv": 1.0,
    "grad_weight_fm": 1.0,
    "grad_balancer_ema_decay": 0.995,
    "use_amp": True,
    "num_workers": 4,
    "n_steps": 40000,
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

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
def repo_root() -> Path:
    # d = Path.cwd() / "dummy" if is_notebook() else Path(__file__)
    d = Path(__file__)
    assert d.is_absolute(), d
    for d in d.parents:
        if (d / ".git").is_dir():
            return d
    raise RuntimeError("Repository root is not found.")



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

def prepare_training_configs(data_dir, out_dir, resume=False, config=None) -> tuple[dict, Path, Path, bool]:
    # data_dir, out_dir は config ファイルでもコマンドライン引数でも指定でき、
    # コマンドライン引数が優先される。
    # 各種ファイルパスを相対パスで指定した場合、config ファイルでは
    # リポジトリルートからの相対パスとなるが、コマンドライン引数では
    # カレントディレクトリからの相対パスとなる。

    # parser = argparse.ArgumentParser()
    # # fmt: off
    # parser.add_argument("-d", "--data_dir", default="data\model_2", type=Path, help="directory containing the training data")
    # parser.add_argument("-o", "--out_dir", default="trained_models\model_2", type=Path, help="output directory")
    # parser.add_argument("-r", "--resume", action="store_true", help="resume training")
    # parser.add_argument("-c", "--config", type=Path, help="path to the config file")
    # # fmt: on
    # args = parser.parse_args()

    # config
    if config is None:
        h = deepcopy(dict_default_hparams)
    else:
        with open(config, encoding="utf-8") as f:
            h = json.load(f)
    for key in dict_default_hparams.keys():
        if key not in h:
            h[key] = dict_default_hparams[key]
            warnings.warn(
                f"{key} is not specified in the config file. Using the default value."
            )
    # data_dir
    if data_dir is not None:
        in_wav_dataset_dir = data_dir
    elif "data_dir" in h:
        in_wav_dataset_dir = repo_root() / Path(h["data_dir"])
        del h["data_dir"]
    else:
        raise ValueError(
            "data_dir must be specified. "
            "For example `python3 beatrice_trainer -d my_training_data_dir -o my_output_dir`."
        )
    # out_dir
    if out_dir is not None:
        out_dir = out_dir
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
    resume = resume
    return h, in_wav_dataset_dir, out_dir, resume

