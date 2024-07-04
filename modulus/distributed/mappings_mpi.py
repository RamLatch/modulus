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

import torch
from mpi4py import MPI
import torch.distributed

from modulus.distributed.utils_mpi import (
    #_reduce_torch as _reduce,
    _reduce,
    _split,
    #all_gather_v_wrapper_torch as all_gather_v_wrapper,
    all_gather_v_wrapper,
    compute_split_shapes,
)
if torch.distributed.is_initialized():
    from modulus.distributed.utils_mpi import (
        _reduce_torch as _reduce,
        all_gather_v_wrapper_torch as all_gather_v_wrapper,
    )

comm = MPI.COMM_WORLD

class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region"""

    @staticmethod
    def symbolic(graph, input_):  # pragma: no cover
        return input_

    @staticmethod
    def forward(ctx, input_):  # pragma: no cover
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return _reduce(grad_output), None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region"""

    @staticmethod
    def symbolic(graph, input_):  # pragma: no cover
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):  # pragma: no cover
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the chunk corresponding to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_):  # pragma: no cover
        return _split(input_, dim_)

    @staticmethod
    def forward(ctx, input_, dim_):  # pragma: no cover
        ctx.dim = dim_
        ctx.split_shapes = compute_split_shapes(
            input_.shape[dim_], comm.Get_size()
        )
        return _split(input_, dim_)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            all_gather_v_wrapper(
                grad_output,
                ctx.split_shapes,
                ctx.dim
            ),
            None,
            None,
        )


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, shapes_):  # pragma: no cover
        return all_gather_v_wrapper(
            input_, shapes_, dim_
        )

    @staticmethod
    def forward(ctx, input_, dim_, shapes_):  # pragma: no cover
        ctx.dim = dim_
        return all_gather_v_wrapper(
            input_, shapes_, dim_
        )

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            _split(grad_output, ctx.dim),
            None,
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------
def copy_to_parallel_region(input):  # pragma: no cover
    """Copy input"""
    return _CopyToParallelRegion.apply(input)


def reduce_from_parallel_region(input):  # pragma: no cover
    """All-reduce the input from the matmul parallel region."""
    return _ReduceFromParallelRegion.apply(input)


def scatter_to_parallel_region(input, dim):  # pragma: no cover
    """Split the input and keep only the corresponding chuck to the rank."""
    return _ScatterToParallelRegion.apply(input, dim)


def gather_from_parallel_region(input, dim, shapes):  # pragma: no cover
    """Gather the input from matmul parallel region and concatenate."""
    return _GatherFromParallelRegion.apply(input, dim, shapes)
