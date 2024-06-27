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

import logging
from functools import partial
from typing import Any, Tuple, Union

import math
import warnings
import torch
import logging

import torch.nn.functional as F
from dataclasses import dataclass
from modulus.models.meta import ModelMetaData
# distributed stuff
from mpi4py import MPI
import torch.fft
import torch.nn as nn
from torch import Tensor

import modulus
from modulus.distributed.mappings_mpi import (
    copy_to_parallel_region,
    gather_from_parallel_region,
    scatter_to_parallel_region,
    reduce_from_parallel_region,
)
from modulus.distributed.utils_mpi import compute_split_shapes, get_memory_format
# distributed stuff
comm = MPI.COMM_WORLD


logger = logging.getLogger(__name__)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2low-1, 2up-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
    tensor: an n-dimensional `torch.Tensor`
    mean: the mean of the normal distribution
    std: the standard deviation of the normal distribution
    a: the minimum cutoff value
    b: the maximum cutoff value
    Examples:
    >>> w = torch.empty(3, 5)
    >>> o = nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    This is the same as the DropConnect implfor EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in
    a separate paper.
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    Opted for changing the layer and argument names to 'drop path' rather than mix
    DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DistributedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
    ):
        super(DistributedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # get effective embedding size:
        comm_size = comm.Get_size()
        if not (hidden_features % comm_size == 0):
            raise ValueError(
                "Error, hidden_features needs to be divisible by matmul_parallel_size"
            )
        hidden_features_local = hidden_features // comm_size

        # first set of hp
        self.w1 = nn.Parameter(torch.ones(hidden_features_local, in_features, 1, 1))
        self.b1 = nn.Parameter(torch.zeros(hidden_features_local))

        # second set of hp
        self.w2 = nn.Parameter(torch.ones(out_features, hidden_features_local, 1, 1))
        self.b2 = nn.Parameter(torch.zeros(out_features))

        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

        if self.input_is_matmul_parallel:
            self.gather_shapes = compute_split_shapes(
                in_features, comm.Get_size()
            )

        # init weights
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.w1, std=0.02)
        nn.init.constant_(self.b1, 0.0)
        trunc_normal_(self.w2, std=0.02)
        nn.init.constant_(self.b2, 0.0)

    def forward(self, x):
        # gather if input is MP
        if self.input_is_matmul_parallel:
            x = gather_from_parallel_region(
                x, dim=1, shapes=self.gather_shapes
            )

        x = copy_to_parallel_region(x)
        x = F.conv2d(x, self.w1, bias=self.b1)
        x = self.act(x)
        x = self.drop(x)
        x = F.conv2d(x, self.w2, bias=None)
        x = reduce_from_parallel_region(x)
        x = x + torch.reshape(self.b2, (1, -1, 1, 1))
        x = self.drop(x)

        # scatter if output is MP
        if self.output_is_matmul_parallel:
            x = scatter_to_parallel_region(x, dim=1)

        return x


class DistributedPatchEmbed(nn.Module):
    def __init__(
        self,
        inp_shape=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=True,
    ):
        super(DistributedPatchEmbed, self).__init__()

        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.Get_size()

        # compute parameters
        print("inp_shape", inp_shape)
        print("patch_size", patch_size)
        num_patches = (inp_shape[1] // patch_size[1]) * (inp_shape[0] // patch_size[0])
        self.inp_shape = (inp_shape[0], inp_shape[1])
        self.patch_size = patch_size
        self.num_patches = num_patches

        if self.input_parallel:
            if not (in_chans % matmul_comm_size == 0):
                raise ValueError(
                    "Error, the in_chans needs to be divisible by matmul_parallel_size"
                )
            self.in_shapes = compute_split_shapes(
                in_chans, comm.Get_size()
            )

        # get effective embedding size:
        if self.output_parallel:
            if not (embed_dim % matmul_comm_size == 0):
                raise ValueError(
                    "Error, the embed_dim needs to be divisible by matmul_parallel_size"
                )
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(
            in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size
        )

        # make sure we reduce them across rank
        self.proj.weight.is_shared_spatial = True
        self.proj.bias.is_shared_spatial = True

    def forward(self, x):
        if self.input_parallel:
            x = gather_from_parallel_region(
                x, dim=1, shapes=self.in_shapes
            )

        if self.output_parallel:
            x = copy_to_parallel_region(x)

        B, C, H, W = x.shape
        if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
            raise ValueError(
                f"Input input size ({H}*{W}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]})."
            )
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


@torch.jit.script
def compl_mul_add_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    tmp = torch.einsum("bkixys,kiot->stbkoxy", a, b)
    res = (
        torch.stack(
            [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
        )
        + c
    )
    return res


@torch.jit.script
def compl_mul_add_fwd_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    tmp = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = tmp + cc
    return torch.view_as_real(res)


class DistributedAFNO2D(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
    ):
        super(DistributedAFNO2D, self).__init__()
        if not (hidden_size % num_blocks == 0):
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )

        # get comm sizes:
        matmul_comm_size = comm.Get_size()

        self.fft_handle = torch.fft.rfft2
        self.ifft_handle = torch.fft.irfft2

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        if not (self.num_blocks % matmul_comm_size == 0):
            raise ValueError(
                "Error, num_blocks needs to be divisible by matmul_parallel_size"
            )
        self.num_blocks_local = self.num_blocks // matmul_comm_size
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        use_complex_mult = False
        self.mult_handle = (
            compl_mul_add_fwd_c if use_complex_mult else compl_mul_add_fwd
        )

        # model parallelism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # new
        # these weights need to be synced across all spatial ranks!
        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size,
                self.block_size * self.hidden_size_factor,
                2,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                1,
                1,
                2,
            )
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                self.block_size,
                2,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks_local, self.block_size, 1, 1, 2)
        )

        # make sure we reduce them across rank
        self.w1.is_shared_spatial = True
        self.b1.is_shared_spatial = True
        self.w2.is_shared_spatial = True
        self.b2.is_shared_spatial = True

    def forward(self, x):
        if not self.input_is_matmul_parallel:
            # distribute data
            num_chans = x.shape[1]
            x = scatter_to_parallel_region(x, dim=1)

        # bias
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        x = self.fft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.view(B, self.num_blocks_local, self.block_size, H, W // 2 + 1)

        # new
        x = torch.view_as_real(x)
        o2 = torch.zeros(x.shape, device=x.device)

        o1 = F.relu(
            self.mult_handle(
                x[
                    :,
                    :,
                    :,
                    total_modes - kept_modes : total_modes + kept_modes,
                    :kept_modes,
                    :,
                ],
                self.w1,
                self.b1,
            )
        )
        o2[
            :, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :
        ] = self.mult_handle(o1, self.w2, self.b2)

        # finalize
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.type(dtype) + bias

        # gather
        if not self.output_is_matmul_parallel:
            gather_shapes = compute_split_shapes(
                num_chans, comm.Get_size()
            )
            x = gather_from_parallel_region(
                x, dim=1, shapes=gather_shapes
            )

        return x

class DistributedBlock(nn.Module):
    def __init__(
        self,
        h,
        w,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
    ):
        super(DistributedBlock, self).__init__()

        # model parallelism
        # matmul parallelism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # norm layer
        self.norm1 = norm_layer((h, w))

        # filter
        self.filter = DistributedAFNO2D(
            dim,
            num_blocks,
            sparsity_threshold,
            hard_thresholding_fraction,
            input_is_matmul_parallel=True,
            output_is_matmul_parallel=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm2 = norm_layer((h, w))

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DistributedMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            input_is_matmul_parallel=True,
            output_is_matmul_parallel=True,
        )
        self.double_skip = double_skip

    def forward(self, x):
        scatter_shapes = compute_split_shapes(
                x.shape[1], comm.Get_size()
            )
        if not self.input_is_matmul_parallel:
            
            x = scatter_to_parallel_region(x, dim=1)

        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual

        if not self.output_is_matmul_parallel:
            x = gather_from_parallel_region(
                x, dim=1, shapes=scatter_shapes
            )

        return x


class DistributedAFNONet(nn.Module):
    def __init__(
        self,
        inp_shape=(720, 1440),
        patch_size=(16, 16),
        in_chans=2,
        out_chans=2,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
    ):
        super().__init__()

        # comm sizes
        matmul_comm_size = comm.Get_size()

        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        print("pre patch embed", "inp_shape",inp_shape, "patch_size", patch_size)
        self.patch_embed = DistributedPatchEmbed(
            inp_shape=inp_shape,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            input_is_matmul_parallel=self.input_is_matmul_parallel,
            output_is_matmul_parallel=True,
        )
        num_patches = self.patch_embed.num_patches

        # original: x = B, H*W, C
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # new: x = B, C, H*W
        self.embed_dim_local = self.embed_dim // matmul_comm_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim_local, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]

        # add blocks
        blks = []
        for i in range(0, depth):
            input_is_matmul_parallel = True  # if i > 0 else False
            output_is_matmul_parallel = True if i < (depth - 1) else False
            blks.append(
                DistributedBlock(
                    h=self.h,
                    w=self.w,
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    input_is_matmul_parallel=input_is_matmul_parallel,
                    output_is_matmul_parallel=output_is_matmul_parallel,
                )
            )
        self.blocks = nn.ModuleList(blks)

        # head
        if self.output_is_matmul_parallel:
            self.out_chans_local = (
                self.out_chans + matmul_comm_size - 1
            ) // matmul_comm_size
        else:
            self.out_chans_local = self.out_chans
        self.head = nn.Conv2d(
            self.embed_dim,
            self.out_chans_local * self.patch_size[0] * self.patch_size[1],
            1,
            bias=False,
        )
        self.synchronized_head = False

        # init weights
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # reshape
        x = x.reshape(B, self.embed_dim_local, self.h, self.w)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        # fw pass on features
        x = self.forward_features(x)
        # be careful if head is distributed
        if self.output_is_matmul_parallel:
            x = copy_to_parallel_region(x)
        else:
            if not self.synchronized_head:
                # If output is not model parallel, synchronize all GPUs params for head
                for param in self.head.parameters():
                    param_data = param.data.cpu().numpy()
                    comm.Bcast(param_data, root=0)
                    param.data = torch.from_numpy(param_data).to(param.device)
                self.synchronized_head = True

        x = self.head(x)

        # new: B, C, H, W
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(
            b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1])
        )

        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "AFNOMPI"
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

class DistributedAFNOMPI(modulus.Module):
    """Distributed Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    inp_shape : Tuple[int, int]
        Input image dimensions (height, width)
    in_channels : int
        Number of input channels
    out_channels: Union[int, Any], optional
        Number of outout channels, by default in_channels
    patch_size : int, optional
        Size of image patchs, by default 16
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    num_blocks : int, optional
        Number of blocks in the frequency weight matrices, by default 4
    channel_parallel_inputs : bool, optional
        Are the inputs sharded along the channel dimension, by default False
    channel_parallel_outputs : bool, optional
        Should the outputs be sharded along the channel dimension, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size, H, W]`
    - Output variable tensor shape: :math:`[N, size, H, W]`

    Example
    -------
    >>> # from modulus.distributed import DistributedManager
    >>> # DistributedManager.initialize()
    >>> # model = modulus.models.afno.DistributedAFNO((64, 64), 2)
    >>> # input = torch.randn(20, 2, 64, 64)
    >>> # output = model(input)
    """

    def __init__(
        self,
        inp_shape: Tuple[int, int],
        in_channels: int,
        out_channels: Union[int, Any] = None,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_blocks: int = 4,
        channel_parallel_inputs: bool = False,
        channel_parallel_outputs: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())

        out_channels = out_channels or in_channels

        

        comm_size = comm.Get_size()
        if channel_parallel_inputs:
            if not (in_channels % comm_size == 0):
                raise ValueError(
                    "Error, in_channels needs to be divisible by model_parallel size"
                )

        self._impl = DistributedAFNONet(
            inp_shape=inp_shape,
            patch_size=(patch_size, patch_size),
            in_chans=in_channels,
            out_chans=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
            input_is_matmul_parallel=False,
            output_is_matmul_parallel=False,
        )

    def forward(self, in_vars: Tensor) -> Tensor:
        return self._impl(in_vars)
