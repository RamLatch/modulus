# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
REPLICATE = False
if REPLICATE:
    import random
    import numpy as np
    import torch
    import pickle
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(threshold=10_000)


from dataclasses import dataclass
from functools import partial
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import modulus  # noqa: F401 for docs
import modulus.models.layers.fft as fft

from ..meta import ModelMetaData
from ..module import Module

Tensor = torch.Tensor
BLOCK_DEBUG = 0
dumps = 0
debugpath = "/hkfs/work/workspace/scratch/ie5012-MA/debug"

class AFNOMlp(nn.Module):
    """Fully-connected Multi-layer perception used inside AFNO

    Parameters
    ----------
    in_features : int
        Input feature size
    latent_features : int
        Latent feature size
    out_features : int
        Output feature size
    activation_fn :  nn.Module, optional
        Activation function, by default nn.GELU
    drop : float, optional
        Drop out rate, by default 0.0
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        activation_fn: nn.Module = nn.GELU(),
        drop: float = 0.0,
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super().__init__()
        self.fc1 = nn.Linear(in_features, latent_features)
        self.act = activation_fn
        self.fc2 = nn.Linear(latent_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        global dumps
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNOMlp_Input.pkl", "wb"))
            # try:    print("AFNOMlp x:",x.detach().cpu().numpy())
            # except: print("AFNOMlp x:",x)
        x = self.fc1(x)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNOMlp_fc1.pkl", "wb"))
            # try:    print("AFNOMlp fc1:",x.detach().cpu().numpy())
            # except: print("AFNOMlp fc1:",x)
        x = self.act(x)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNOMlp_act.pkl", "wb"))
            # try:    print("AFNOMlp act:",x.detach().cpu().numpy())
            # except: print("AFNOMlp act:",x)
        x = self.drop(x)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNOMlp_drop.pkl", "wb"))
            # try:    print("AFNOMlp drop:",x.detach().cpu().numpy())
            # except: print("AFNOMlp drop:",x)
        x = self.fc2(x)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNOMlp_fc2.pkl", "wb"))
            # try:    print("AFNOMlp fc2:",x.detach().cpu().numpy())
            # except: print("AFNOMlp fc2:",x)
        x = self.drop(x)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNOMlp_drop2.pkl", "wb"))
            # try:    print("AFNOMlp return:",x.detach().cpu().numpy())
            # except: print("AFNOMlp return:",x)
        return x


class AFNO2DLayer(nn.Module):
    """AFNO spectral convolution layer

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    hidden_size_factor : int, optional
        Factor to increase spectral features by after weight multiplication, by default 1
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1,
        hidden_size_factor: int = 1,
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super().__init__()
        if not (hidden_size % num_blocks == 0):
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor,))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size,))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x: Tensor) -> Tensor:
        global dumps
        bias = x
        dtype = x.dtype
        x = x.float()
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_Input.pkl", "wb"))
            # try:    print("AFNO2DLayer x:",x.detach().cpu().numpy())
            # except: print("AFNO2DLayer x:",x)
        B, H, W, C = x.shape
        # Using ONNX friendly FFT functions
        x = fft.rfft2(x, dim=(1, 2), norm="ortho")
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_rfft2.pkl", "wb"))
            # try:    print("AFNO2DLayer rfft2:",x.detach().cpu().numpy())
            # except: print("AFNO2DLayer rfft2:",x)
        x_real, x_imag = fft.real(x), fft.imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_x_real.pkl", "wb"))
            # try:    print("AFNO2DLayer x_real:",x_real.detach().cpu().numpy())
            # except: print("AFNO2DLayer x_real:",x_real)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_x_imag.pkl", "wb"))
            # try:    print("AFNO2DLayer x_imag:",x_imag.detach().cpu().numpy())
            # except: print("AFNO2DLayer x_imag:",x_imag)

        o1_real = torch.zeros([B,H,W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],device=x.device)
        o1_imag = torch.zeros([B,H,W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],device=x.device)
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum("nyxbi,bio->nyxbo",x_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w1[0]) -
            torch.einsum("nyxbi,bio->nyxbo",x_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w1[1]) +
            self.b1[0]
        )
        if REPLICATE:
            # dumps += 1
            pickle.dump(o1_real, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_o1_real.pkl", "wb"))
            # try:    print("AFNO2DLayer o1_real:",o1_real.detach().cpu().numpy())
            # except: print("AFNO2DLayer o1_real:",o1_real)
        o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = F.relu(
            torch.einsum("nyxbi,bio->nyxbo",x_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w1[0]) +
            torch.einsum("nyxbi,bio->nyxbo",x_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w1[1]) +
            self.b1[1]
        )
        if REPLICATE:
            # dumps += 1
            pickle.dump(o1_imag, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_o1_imag.pkl", "wb"))
            # try:    print("AFNO2DLayer o1_imag:",o1_imag.detach().cpu().numpy())
            # except: print("AFNO2DLayer o1_imag:",o1_imag)
        o2[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 0] = (
            torch.einsum("nyxbi,bio->nyxbo",o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w2[0]) -
            torch.einsum("nyxbi,bio->nyxbo",o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w2[1]) +
            self.b2[0]
        )
        if REPLICATE:
            dumps += 1
            pickle.dump(o2[..., 0], open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_o2_real.pkl", "wb"))
            # try:    print("AFNO2DLayer o2_real:",o2[..., 0].detach().cpu().numpy())
            # except: print("AFNO2DLayer o2_real:",o2[..., 0])
        o2[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 1] = (
            torch.einsum("nyxbi,bio->nyxbo",o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w2[0]) +
            torch.einsum("nyxbi,bio->nyxbo",o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes],self.w2[1]) +
            self.b2[1]
        )
        if REPLICATE:
            # dumps += 1
            pickle.dump(o2[..., 1], open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_o2_imag.pkl", "wb"))
            # try:    print("AFNO2DLayer o2_imag:",o2[..., 1].detach().cpu().numpy())
            # except: print("AFNO2DLayer o2_imag:",o2[..., 1])
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = fft.view_as_complex(x)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_x_complex.pkl", "wb"))
            # try:    print("AFNO2DLayer x_complex:",x.detach().cpu().numpy())
            # except: print("AFNO2DLayer x_complex:",x)
        # TODO(akamenev): replace the following branching with
        # a one-liner, something like x.reshape(..., -1).squeeze(-1),
        # but this currently fails during ONNX export.
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        # Using ONNX friendly FFT functions
        x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_irfft2.pkl", "wb"))
            # try:    print("AFNO2DLayer irfft2:",x.detach().cpu().numpy())
            # except: print("AFNO2DLayer irfft2:",x)
        x = x.type(dtype)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO2DLayer_return.pkl", "wb"))
            # try:    print("AFNO2DLayer return:",x+bias.detach().cpu().numpy())
            # except: print("AFNO2DLayer return:",x+bias)
        return x + bias


class Block(nn.Module):
    """AFNO block, spectral convolution and MLP

    Parameters
    ----------
    embed_dim : int
        Embedded feature dimensionality
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    mlp_ratio : float, optional
        Ratio of MLP latent variable size to input feature size, by default 4.0
    drop : float, optional
        Drop out rate in MLP, by default 0.0
    activation_fn: nn.Module, optional
        Activation function used in MLP, by default nn.GELU
    norm_layer : nn.Module, optional
        Normalization function, by default nn.LayerNorm
    double_skip : bool, optional
        Residual, by default True
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        activation_fn: nn.Module = nn.GELU(),
        norm_layer: nn.Module = nn.LayerNorm,
        double_skip: bool = True,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = AFNO2DLayer(
            embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        self.mlp = AFNOMlp(
            in_features=embed_dim,
            latent_features=mlp_latent_dim,
            out_features=embed_dim,
            activation_fn=activation_fn,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x: Tensor) -> Tensor:
        global dumps
        residual = x
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_Block_{BLOCK_DEBUG}_Input.pkl", "wb"))
            # try:    print(f"Block {BLOCK_DEBUG} residual:",x.detach().cpu().numpy())
            # except: print(f"Block {BLOCK_DEBUG} residual:",x)
        x = self.norm1(x)
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_Block_{BLOCK_DEBUG}_norm1.pkl", "wb"))
            # try:    print(f"Block {BLOCK_DEBUG} norm1:",x.detach().cpu().numpy())
            # except: print(f"Block {BLOCK_DEBUG} norm1:",x)
        x = self.filter(x)
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_Block_{BLOCK_DEBUG}_filter.pkl", "wb"))
            # try:    print(f"Block {BLOCK_DEBUG} filter:",x.detach().cpu().numpy())
            # except: print(f"Block {BLOCK_DEBUG} filter:",x)

        if self.double_skip:
            x = x + residual
            residual = x
            if REPLICATE:
                dumps += 1
                pickle.dump(x, open(f"{debugpath}/{dumps:03d}_Block_{BLOCK_DEBUG}_double_skip.pkl", "wb"))
                # try:    print(f"Block {BLOCK_DEBUG} double skip:",x.detach().cpu().numpy())
                # except: print(f"Block {BLOCK_DEBUG} double skip:",x)

        x = self.norm2(x)
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_Block_{BLOCK_DEBUG}_norm2.pkl", "wb"))
            # try:    print(f"Block {BLOCK_DEBUG} norm2:",x.detach().cpu().numpy())
            # except: print(f"Block {BLOCK_DEBUG} norm2:",x)
        x = self.mlp(x)
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_Block_{BLOCK_DEBUG}_mlp.pkl", "wb"))
            # try:    print(f"Block {BLOCK_DEBUG} mlp:",x.detach().cpu().numpy())
            # except: print(f"Block {BLOCK_DEBUG} mlp:",x)
        x = x + residual
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_Block_{BLOCK_DEBUG}_return.pkl", "wb"))
            # try:    print(f"Block {BLOCK_DEBUG} return:",x.detach().cpu().numpy())
            # except: print(f"Block {BLOCK_DEBUG} return:",x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer

    Converts 2D patch into a 1D vector for input to AFNO

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            torch.random.set_rng_state(pickle.load(open(f"{debugpath}/torchrandomstate.pkl", "rb")))
            torch.cuda.random.set_rng_state_all(pickle.load(open(f"{debugpath}/cudarandomstate.pkl", "rb")))

        super().__init__()
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        num_patches = (inp_shape[1] // patch_size[1]) * (inp_shape[0] // patch_size[0])
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        if REPLICATE:
            #if os.path.exists(f"{debugpath}/Conv2d.pkl"):
            #    self.proj = pickle.load(open(f"{debugpath}/Conv2d.pkl", "rb"))
            #else:
                self.proj = nn.Conv2d(
                    in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
                )
                print("self.proj.weight",self.proj.weight)
                print("self.proj.bias",self.proj.bias)
                pickle.dump(self.proj.weight, open(f"{debugpath}/Patchembed_Conv2d_weight.pkl", "wb"))
                pickle.dump(self.proj.bias, open(f"{debugpath}/Patchembed_Conv2d_bias.pkl", "wb"))
        else:
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        global dumps
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_PatchEmbed_Input.pkl", "wb"))
            # try:    print("PatchEmbed x:",x.detach().cpu().numpy())
            # except: print("PatchEmbed x:",x)
        if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]})."
            )
        if REPLICATE:
            self.proj.weight = nn.Parameter(pickle.load(open(f"{debugpath}/Patchembed_Conv2d_weight.pkl", "rb")).type_as(self.proj.weight))
            self.proj.bias = nn.Parameter(pickle.load(open(f"{debugpath}/Patchembed_Conv2d_bias.pkl", "rb")).type_as(self.proj.bias))
        x = self.proj(x).flatten(2).transpose(1, 2)
        if REPLICATE:
            pickle.dump(self.proj.weight, open(f"{debugpath}/Patchembed_Conv2d_weight_after.pkl", "wb"))
            pickle.dump(self.proj.bias, open(f"{debugpath}/Patchembed_Conv2d_bias_after.pkl", "wb"))
        if REPLICATE:
            # dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_PatchEmbed_return.pkl", "wb"))
            # try:    print("PatchEmbed return:",x.detach().cpu().numpy())
            # except: print("PatchEmbed return:",x)
        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "AFNO"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class AFNO(Module):
    """Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    out_channels: int
        Number of output channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1

    Example
    -------
    >>> model = modulus.models.afno.AFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    Note
    ----
    Reference: Guibas, John, et al. "Adaptive fourier neural operators:
    Efficient token mixers for transformers." arXiv preprint arXiv:2111.13587 (2021).
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        out_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super().__init__(meta=MetaData())
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        if not (
            inp_shape[0] % patch_size[0] == 0 and inp_shape[1] % patch_size[1] == 0
        ):
            raise ValueError(
                f"input shape {inp_shape} should be divisible by patch_size {patch_size}"
            )

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            inp_shape=inp_shape,
            in_channels=self.in_chans,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # What is this for
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {"pos_embed", "cls_token"}

    def forward_features(self, x: Tensor) -> Tensor:
        global BLOCK_DEBUG
        global dumps
        """Forward pass of core AFNO"""
        B = x.shape[0]
        x = self.patch_embed(x)
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_patch_embed.pkl", "wb"))
            # try:    print("Afno patch embed:",x.detach().cpu().numpy())
            # except: print("Afno patch embed:",x)
        x = x + self.pos_embed
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_pos_embed.pkl", "wb"))
            # try:    print("Afno pos embed:",x.detach().cpu().numpy())
            # except: print("Afno pos embed:",x)
        x = self.pos_drop(x)
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_pos_drop.pkl", "wb"))
            # try:    print("Afno pos drop:",x.detach().cpu().numpy())
            # except: print("Afno pos drop:",x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        if REPLICATE:
            #dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_reshape.pkl", "wb"))
            # try:    print("Afno reshape:",x.detach().cpu().numpy())
            # except: print("Afno reshape:",x)
        for blk in self.blocks:
            x = blk(x)
            BLOCK_DEBUG += 1
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_blocks.pkl", "wb"))
            # try:    print("Afno return:",x.detach().cpu().numpy())
            # except: print("Afno return:",x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        global dumps
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_Input.pkl", "wb"))
            # try:    print("Afno forward:",x.detach().cpu().numpy())
            # except: print("Afno forward:",x)
        x = self.forward_features(x)
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_forward_features.pkl", "wb"))
            # try:    print("Afno forward features:",x.detach().cpu().numpy())
            # except: print("Afno forward features:",x)
        x = self.head(x)
        if REPLICATE:
            dumps += 1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_AFNO_head.pkl", "wb"))
            # try:    print("Afno head:",x.detach().cpu().numpy())
            # except: print("Afno head:",x)
        # Correct tensor shape back into [B, C, H, W]
        # [b h w (p1 p2 c_out)]
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        # [b h w p1 p2 c_out]
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        # [b c_out, h, p1, w, p2]
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])
        # [b c_out, (h*p1), (w*p2)]
        if REPLICATE:
            dumps += 1
            pickle.dump(out, open(f"{debugpath}/{dumps:03d}_AFNO_return.pkl", "wb"))
            # try:    print("Afno return:",out.detach().cpu().numpy())
            # except: print("Afno return:",out)
        if REPLICATE: exit(1)
        return out
