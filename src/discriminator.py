
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from typing import BinaryIO, Literal, Optional, Union
from fractions import Fraction

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
