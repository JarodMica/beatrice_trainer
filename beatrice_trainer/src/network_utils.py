import os
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from functools import partial
from typing import BinaryIO, Literal, Optional, Union

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