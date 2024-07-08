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
PRINTGRAPH = True
if PRINTGRAPH:
    from torchviz import make_dot
    GRAPHEPOCH = 0
import logging, os
from functools import partial
from typing import Any, Tuple, Union

import math
import warnings
import torch


import torch.distributed
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
dist = torch.distributed
try:
    if not REPLICATE: LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if not REPLICATE: WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if not REPLICATE: WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
except:
    LOCAL_RANK, WORLD_SIZE, WORLD_RANK = None, None, None
BLOCK_DEBUG = 0
logger = logging.getLogger(__name__)
dumps = 0
debugpath = "/hkfs/work/workspace/scratch/ie5012-MA/debug"

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
    global dumps
    if REPLICATE:
        dumps +=1
        pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_drop_path.pkl", "wb"))
        # try:    print("drop_path x:",x.detach().cpu().numpy())
        # except: print("drop_path x:",x)
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    if REPLICATE:
        # dumps +=1
        pickle.dump(output, open(f"{debugpath}/{dumps:03d}_mpi_drop_path_output.pkl", "wb"))
        # try:    print("drop_path output:",output.detach().cpu().numpy())
        # except: print("drop_path output:",output)
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    """

    def __init__(self, drop_prob=None):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        global dumps
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
        comm:MPI.Intracomm=None
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super(DistributedMLP, self).__init__()
        self.comm = comm
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # get effective embedding size:
        comm_size = comm.Get_size() #if not dist.is_initialized() else dist.get_world_size()
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
                in_features, comm_size
            )

        # init weights
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.constant_(self.b1, 0.0)
        torch.nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.constant_(self.b2, 0.0)

    def forward(self, x):
        global dumps
        # gather if input is MP
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp.pkl", "wb"))
            # try:   print("DistributedMLP x:",x.detach().cpu().numpy())
            # except:print("DistributedMLP x:",x)
        if self.comm.Get_size() > 1:
            if self.input_is_matmul_parallel:
                x = gather_from_parallel_region(
                    x, dim=1, shapes=self.gather_shapes
                )
                if REPLICATE:
                    dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_gather.pkl", "wb"))
                    # try:    print("DistributedMLP x after gather if inputMatMulParallel:",x.detach().cpu().numpy())
                    # except: print("DistributedMLP x after gather if inputMatMulParallel:",x)

            x = copy_to_parallel_region(x)
            if REPLICATE:
                dumps +=1
                pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_copy_to_parallel_region.pkl", "wb"))
                # try:    print("DistributedMLP x after copy_to_parallel_region:",x.detach().cpu().numpy())
                # except: print("DistributedMLP x after copy_to_parallel_region:",x)
        x = F.conv2d(x, self.w1, bias=self.b1)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_conv2d.pkl", "wb"))
            # try:    print("DistributedMLP conf2d:",x.detach().cpu().numpy())
            # except: print("DistributedMLP conf2d:",x)
        x = self.act(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_act.pkl", "wb"))
            # try:    print("DistributedMLP act:",x.detach().cpu().numpy())
            # except: print("DistributedMLP act:",x)
        x = self.drop(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_drop.pkl", "wb"))
            # try:    print("DistributedMLP drop:",x.detach().cpu().numpy())
            # except: print("DistributedMLP drop:",x)
        x = F.conv2d(x, self.w2, bias=None)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_conv2d2.pkl", "wb"))
            # try:    print("DistributedMLP conf2d:",x.detach().cpu().numpy())
            # except: print("DistributedMLP conf2d:",x)
        if self.comm.Get_size() > 1:
            x = reduce_from_parallel_region(x)
            if REPLICATE:
                dumps +=1
                pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_reduce_from_parallel_region.pkl", "wb"))
                # try:    print("DistributedMLP x after reduce_from_parallel_region:",x.detach().cpu().numpy())
                # except: print("DistributedMLP x after reduce_from_parallel_region:",x)
        x = x + torch.reshape(self.b2, (1, -1, 1, 1))
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_b2.pkl", "wb"))
            # try:    print("DistributedMLP x after reshape:",x.detach().cpu().numpy())
            # except: print("DistributedMLP x after reshape:",x)
        x = self.drop(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_drop2.pkl", "wb"))
            # try:    print("DistributedMLP drop:",x.detach().cpu().numpy())
            # except: print("DistributedMLP drop:",x)

        # scatter if output is MP
        if self.comm.Get_size() > 1:
            if self.output_is_matmul_parallel:
                x = scatter_to_parallel_region(x, dim=1)
                if REPLICATE:
                    dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_mlp_scatter_to_parallel_region.pkl", "wb"))
                    # try:    print("DistributedMLP x after scatter if outMatMul:",x.detach().cpu().numpy())
                    # except: print("DistributedMLP x after scatter if outMatMul:",x)
        if PRINTGRAPH and GRAPHEPOCH == 0:
            make_dot(x).render(f"{debugpath}/DistributedMLP", format="png")
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
        comm:MPI.Intracomm=None
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            torch.random.set_rng_state(pickle.load(open(f"{debugpath}/torchrandomstate.pkl", "rb")))
            torch.cuda.random.set_rng_state_all(pickle.load(open(f"{debugpath}/cudarandomstate.pkl", "rb")))
        super(DistributedPatchEmbed, self).__init__()
        self.comm = comm
        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.Get_size() #if not dist.is_available() else dist.get_world_size("model_parallel")

        # compute parameters
        # print("inp_shape", inp_shape.detach().cpu().numpy())
        # print("patch_size", patch_size.detach().cpu().numpy())
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
                in_chans, matmul_comm_size
            )

        # get effective embedding size:
        if self.output_parallel:
            if not (embed_dim % matmul_comm_size == 0):
                raise ValueError(
                    "Error, the embed_dim needs to be divisible by matmul_parallel_size"
                )
            out_chans_local = embed_dim // matmul_comm_size
            with open(f"{debugpath}/embed_dim", "a") as f:
                f.write("matmul_comm_size "+str(matmul_comm_size))
        else:
            out_chans_local = embed_dim
        with open(f"{debugpath}/embed_dim", "a") as f:
            f.write("embed_dim "+str(out_chans_local))
        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(
                in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size
        )
        #print("self.proj.weight",self.proj.weight)
        #print("self.proj.bias",self.proj.bias)
        if REPLICATE:
            pickle.dump(self.proj.weight, open(f"{debugpath}/DistPatchembed_Conv2d_weight.pkl", "wb"))
            pickle.dump(self.proj.bias, open(f"{debugpath}/DistPatchembed_Conv2d_bias.pkl", "wb"))
        # make sure we reduce them across rank
        self.proj.weight.is_shared_spatial = True
        self.proj.bias.is_shared_spatial = True

    def forward(self, x: Tensor):
        global dumps
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_patch_embed_Input.pkl", "wb"))
            # try:    print("DistributedPatchEmbed x:",x.detach().cpu().numpy())
            # except: print("DistributedPatchEmbed x:",x)
        if self.comm.Get_size() > 1:
            if self.input_parallel:
                x = gather_from_parallel_region(
                    x, dim=1, shapes=self.in_shapes
                )
                if REPLICATE:
                    # dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_patch_embed_gather.pkl", "wb"))
                    # try:    print("DistributedPatchEmbed x after gather if input_parallel:",x.detach().cpu().numpy())
                    # except: print("DistributedPatchEmbed x after gather if input_parallel:",x)

            if self.output_parallel:
                x = copy_to_parallel_region(x)
                if REPLICATE:
                    # dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_patch_embed_copy_to_parallel_region.pkl", "wb"))
                    # try:    print("DistributedPatchEmbed x after copy_to_parallel_region:",x.detach().cpu().numpy())
                    # except: print("DistributedPatchEmbed x after copy_to_parallel_region:",x)

        B, C, H, W = x.shape
        if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
            raise ValueError(
                f"Input input size ({H}*{W}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]})."
            )
        # new: B, C, H*W
        if REPLICATE:
            self.proj.weight = nn.Parameter(pickle.load(open(f"{debugpath}/DistPatchembed_Conv2d_weight.pkl", "rb")).type_as(self.proj.weight))
            self.proj.bias = nn.Parameter(pickle.load(open(f"{debugpath}/DistPatchembed_Conv2d_bias.pkl", "rb")).type_as(self.proj.bias))
            x= self.proj(x).flatten(2).transpose(1, 2)
        else: x = self.proj(x).flatten(2)
        
        if REPLICATE:
            pickle.dump(self.proj.weight, open(f"{debugpath}/DistPatchembed_Conv2d_weight_after.pkl", "wb"))
            pickle.dump(self.proj.bias, open(f"{debugpath}/DistPatchembed_Conv2d_bias_after.pkl", "wb"))
        # x = self.proj(x).flatten(2).transpose(1, 2)
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_patch_embed_return.pkl", "wb"))
            # try:    print("DistributedPatchEmbed return:",x.detach().cpu().numpy())
            # except: print("DistributedPatchEmbed return:",x)
        if PRINTGRAPH and GRAPHEPOCH == 0:
            make_dot(x).render(f"{debugpath}/DistributedPatchEmbed", format="png")
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
        comm:MPI.Intracomm=None
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super(DistributedAFNO2D, self).__init__()
        if not (hidden_size % num_blocks == 0):
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )
        self.comm = comm
        # get comm sizes:
        matmul_comm_size = comm.Get_size() #if not dist.is_available() else dist.get_world_size("model_parallel")
        self.matmul_comm_size = matmul_comm_size
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
        global dumps
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_Input.pkl", "wb"))
            # try:    print("DistributedAFNO2D x:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D x:",x)
        if self.comm.Get_size() > 1:
            if not self.input_is_matmul_parallel:
                # distribute data
                num_chans = x.shape[1]
                x = scatter_to_parallel_region(x, dim=1)
                if REPLICATE:
                    # dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_scatter_to_parallel_region.pkl", "wb"))
                    # try:    print("DistributedAFNO2D x after scatter if not inputMatMulParallel:",x.detach().cpu().numpy())
                    # except: print("DistributedAFNO2D x after scatter if not inputMatMulParallel:",x)

        # bias
        bias = x

        dtype = x.dtype
        x = x.float()
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_float.pkl", "wb"))
            # try:    print("DistributedAFNO2D x after float:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D x after float:",x)
        B, C, H, W = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        x = self.fft_handle(x, (H, W), (-2, -1), "ortho")
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_fft_handle.pkl", "wb"))
            # try:    print("DistributedAFNO2D x after fft_handle:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D x after fft_handle:",x)
        x = x.view(B, self.num_blocks_local, self.block_size, H, W // 2 + 1)
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_view.pkl", "wb"))
            # try:    print("DistributedAFNO2D x after view:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D x after view:",x)

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
        if REPLICATE:
            # dumps +=1
            pickle.dump(o1, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_o1.pkl", "wb"))
            # try:    print("DistributedAFNO2D o1:",o1.detach().cpu().numpy())
            # except: print("DistributedAFNO2D o1:",o1)
        o2[
            :, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :
        ] = self.mult_handle(o1, self.w2, self.b2)
        if REPLICATE:
            # dumps +=1
            pickle.dump(o2, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_o2.pkl", "wb"))
            # try:    print("DistributedAFNO2D o2:",o2.detach().cpu().numpy())
            # except: print("DistributedAFNO2D o2:",o2)
        # finalize
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_softshrink.pkl", "wb"))
            # try:    print("DistributedAFNO2D x after softshrink:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D x after softshrink:",x)
        x = x.reshape(B, C, H, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), "ortho")
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_ifft_handle.pkl", "wb"))
            # try:    print("DistributedAFNO2D x after ifft_handle:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D x after ifft_handle:",x)
        x = x.type(dtype) + bias
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_type.pkl", "wb"))
            # try:    print("DistributedAFNO2D x after type:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D x after type:",x)

        # gather
        if self.comm.Get_size() > 1:
            if not self.output_is_matmul_parallel:
                gather_shapes = compute_split_shapes(
                    num_chans, self.matmul_comm_size
                )
                x = gather_from_parallel_region(
                    x, dim=1, shapes=gather_shapes
                )
                if REPLICATE:
                    # dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_gather_from_parallel_region.pkl", "wb"))
                    # try:    print("DistributedAFNO2D x after gather if not outputMatMulParallel:",x.detach().cpu().numpy())
                    # except: print("DistributedAFNO2D x after gather if not outputMatMulParallel:",x)
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afno2d_return.pkl", "wb"))
            # try:    print("DistributedAFNO2D return:",x.detach().cpu().numpy())
            # except: print("DistributedAFNO2D return:",x)
        if PRINTGRAPH and GRAPHEPOCH == 0 and BLOCK_DEBUG == 0:
            make_dot(x).render(f"{debugpath}/DistributedAFNO2D", format="png")
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
        comm:MPI.Intracomm=None
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super(DistributedBlock, self).__init__()

        # model parallelism
        self.world_size=comm.Get_size() #if not dist.is_available() else dist.get_world_size("model_parallel")
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
            comm=comm
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
            comm=comm
        )
        self.double_skip = double_skip

    def forward(self, x):
        global dumps
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}_Input.pkl", "wb"))
            # try:    print(f"DistributedBlock {BLOCK_DEBUG} x:",x.detach().cpu().numpy())
            # except: print(f"DistributedBlock {BLOCK_DEBUG} x:",x)
        scatter_shapes = compute_split_shapes(
                x.shape[1], self.world_size
            )
        if self.world_size > 1:
            if not self.input_is_matmul_parallel:
                
                x = scatter_to_parallel_region(x, dim=1)
                if REPLICATE:
                    # dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}scatter_to_parallel_region.pkl", "wb"))
                    # try:    print(f"DistributedBlock {BLOCK_DEBUG} x after scatter if not inputMatMulParallel:",x.detach().cpu().numpy())
                    # except: print(f"DistributedBlock {BLOCK_DEBUG} x after scatter if not inputMatMulParallel:",x)

        residual = x
        x = self.norm1(x)
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}norm1.pkl", "wb"))
            # try:    print(f"DistributedBlock {BLOCK_DEBUG} norm1:",x.detach().cpu().numpy())
            # except: print(f"DistributedBlock {BLOCK_DEBUG} norm1:",x)
        x = self.filter(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}filter.pkl", "wb"))
            # try:    print(f"DistributedBlock {BLOCK_DEBUG} filter:",x.detach().cpu().numpy())
            # except: print(f"DistributedBlock {BLOCK_DEBUG} filter:",x)

        if self.double_skip:
            x = x + residual
            residual = x
            if REPLICATE:
                dumps +=1
                pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}double_skip.pkl", "wb"))
                # try:    print(f"DistributedBlock {BLOCK_DEBUG} double_skip:",x.detach().cpu().numpy())
                # except: print(f"DistributedBlock {BLOCK_DEBUG} double_skip:",x)

        x = self.norm2(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}norm2.pkl", "wb"))
            # try:    print(f"DistributedBlock {BLOCK_DEBUG} norm2:",x.detach().cpu().numpy())
            # except: print(f"DistributedBlock {BLOCK_DEBUG} norm2:",x)
        x = self.mlp(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}mlp.pkl", "wb"))
            # try:    print(f"DistributedBlock {BLOCK_DEBUG} mlp:",x.detach().cpu().numpy())
            # except: print(f"DistributedBlock {BLOCK_DEBUG} mlp:",x)
        x = self.drop_path(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}drop_path.pkl", "wb"))
            # try:    print(f"DistributedBlock {BLOCK_DEBUG} drop_path:",x.detach().cpu().numpy())
            # except: print(f"DistributedBlock {BLOCK_DEBUG} drop_path:",x)
        x = x + residual
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}residual.pkl", "wb"))
            # try:    print(f"DistributedBlock {BLOCK_DEBUG} x after residual:",x.detach().cpu().numpy())
            # except: print(f"DistributedBlock {BLOCK_DEBUG} x after residual:",x)
        
        if self.world_size > 1:
            if not self.output_is_matmul_parallel:
                x = gather_from_parallel_region(
                    x, dim=1, shapes=scatter_shapes
                )
                if REPLICATE:
                    dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_block_{BLOCK_DEBUG}gather_from_parallel_region.pkl", "wb"))
                    # try:    print(f"DistributedBlock {BLOCK_DEBUG} x after gather if not outputMatMulParallel:",x.detach().cpu().numpy())
                    # except: print(f"DistributedBlock {BLOCK_DEBUG} x after gather if not outputMatMulParallel:",x)
        if PRINTGRAPH and GRAPHEPOCH == 0 and BLOCK_DEBUG == 0:
            make_dot(x).render(f"{debugpath}/DistributedBlock", format="png")
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
        comm:MPI.Intracomm=None,
    ):
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super().__init__()
        self.comm=comm
        self.rank = comm.Get_rank() #if not dist.is_initialized() else dist.get_rank("model_parallel")
        self.world_size = comm.Get_size() #if not dist.is_initialized() else dist.get_world_size("model_parallel")

        # comm sizes
        matmul_comm_size = self.world_size

        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # print("pre patch embed", "inp_shape",inp_shape, "patch_size", patch_size.detach().cpu().numpy())
        self.patch_embed = DistributedPatchEmbed(
            inp_shape=inp_shape,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            input_is_matmul_parallel=self.input_is_matmul_parallel,
            output_is_matmul_parallel=True,
            comm=comm,
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
                    comm=self.comm,
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
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        global dumps
        global BLOCK_DEBUG
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_forward_features_input.pkl", "wb"))
            # try:    print("DistributedAFNONet forward_features x:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet forward_features x:",x)
        B = x.shape[0]

        x = self.patch_embed(x)
        if PRINTGRAPH and GRAPHEPOCH == 0:
            make_dot(x).render(f"{debugpath}/DistributedPatch", format="png")
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_patch_embed.pkl", "wb"))
            # try:    print("DistributedAFNONet patch_embed:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet patch_embed:",x)
        x = x + self.pos_embed#.transpose(1, 2)
        if PRINTGRAPH and GRAPHEPOCH == 0:
            make_dot(x,dict(list(self.pos_embed.named_parameters()))).render(f"{debugpath}/DistributedPose", format="png")

        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_pos_embed.pkl", "wb"))
            # try:    print("DistributedAFNONet pos_embed:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet pos_embed:",x)
        x = self.pos_drop(x)
        if PRINTGRAPH and GRAPHEPOCH == 0:
            make_dot(x).render(f"{debugpath}/DistributedPoseDrop", format="png")

        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_pos_drop.pkl", "wb"))
            # try:    print("DistributedAFNONet pos_drop:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet pos_drop:",x)

        # reshape
        x = x.reshape(B, self.embed_dim_local, self.h, self.w)
        if REPLICATE:
            # dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_reshape.pkl", "wb"))
            # try:    print("DistributedAFNONet x after reshape:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet x after reshape:",x)

        for blk in self.blocks:
            x = blk(x)
            BLOCK_DEBUG += 1
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_blocks.pkl", "wb"))
            # try:    print("DistributedAFNONet forward_features return x:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet forward_features return x:",x)
        return x

    def forward(self, x):
        global dumps
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_Input.pkl", "wb"))
            # try:    print("DistributedAFNONet x:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet x:",x)
        # fw pass 
        x = self.forward_features(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_forward_features.pkl", "wb"))
            # try:    print("DistributedAFNONet x after copy_to if outMatMul:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet x after copy_to if outMatMul:",x)
        # be careful if head is distributed
        if self.comm.Get_size() > 1:
            if self.output_is_matmul_parallel:
                x = copy_to_parallel_region(x)
                if REPLICATE:
                    # dumps +=1
                    pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_copy_to_parallel_region.pkl", "wb"))
                    # try:    print("DistributedAFNONet x after copy_to if outMatMul:",x.detach().cpu().numpy())
                    # except: print("DistributedAFNONet x after copy_to if outMatMul:",x)
            else:
                if not self.synchronized_head:
                    # If output is not model parallel, synchronize all GPUs params for head
                    for param in self.head.parameters():
                        param_data = param.data.cpu().numpy()
                        self.comm.Bcast(param_data, root=0) #if not dist.is_initialized() else dist.broadcast(param_data, 0, "model_parallel")
                        param.data = torch.from_numpy(param_data).to(param.device)
                    self.synchronized_head = True

        x = self.head(x)
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_head.pkl", "wb"))
            # try:    print("DistributedAFNONet head:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet head:",x)

        # new: B, C, H, W
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(
            b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1])
        )
        if REPLICATE:
            dumps +=1
            pickle.dump(x, open(f"{debugpath}/{dumps:03d}_mpi_distributed_afnonet_return.pkl", "wb"))
            # try:    print("DistributedAFNONet return:",x.detach().cpu().numpy())
            # except: print("DistributedAFNONet return:",x)
        if REPLICATE: exit(1)
        global GRAPHEPOCH
        if PRINTGRAPH and GRAPHEPOCH == 0:
            make_dot(x).render(f"{debugpath}/DistributedFull",format="png")
            GRAPHEPOCH = 1
        # exit(1)
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
        comm:MPI.Intracomm=None,
    ) -> None:
        if REPLICATE:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        super().__init__(meta=MetaData())

        out_channels = out_channels or in_channels

        

        comm_size = comm.Get_size() #if not dist.is_initialized() else dist.get_world_size("model_parallel")
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
            comm=comm,
        )

    def forward(self, in_vars: Tensor) -> Tensor:
        return self._impl(in_vars)
