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
        comm = MPI.COMM_WORLD
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
def split_tensor_along_dim(tensor, dim, num_chunks):
    if num_chunks == 1:
        return [tensor.shape[dim]]
    chunk_size = (tensor.shape[dim] + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, tensor.shape[dim] - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        chunk_size = tensor.shape[dim] // num_chunks
        last_chunk_size = tensor.shape[dim] - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]
    tensor_list = torch.split(tensor, sections, dim=dim)
    return tensor_list

#an torch autograd function that handles forward and backward pass For MPI Scatter
class ScatterFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
        return g.op("Scatter", input, rank=rank, size=comm_size)
    @staticmethod
    def forward(ctx: torch.autograd.function._ContextMethodMixin, input: torch.Tensor, dim_):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
        ctx.save_for_backward(input)
        ctx.dim_ = dim_
        input_format= torch.channels_last   if input.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
        output=comm.scatter(split_tensor_along_dim(input.clone().detach(), dim_, comm_size))
        output = output.contiguous(memory_format=input_format)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function._ContextMethodMixin, grad_output):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
        input, = ctx.saved_tensors
        output=comm.allgather(grad_output.clone().detach())
        #combine allgathered tensors
        output=torch.cat(output,ctx.dim_)
        return (output, None)
    
class AllgatherVFunction(torch.autograd.Function):
    
    @staticmethod
    def symbolic(g, input):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
        return g.op("AllgatherV", input, rank=rank, size=comm_size)
    @staticmethod
    def forward(ctx: torch.autograd.function._ContextMethodMixin, input: torch.Tensor, dim_):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
        ctx.dim = dim_
        ctx.save_for_backward(input)
        comm.barrier()
        print(f"testing: {input[0][0].shape}")
        if rank == 0:
            comm.send(input[0][0], dest=1)
        elif rank == 1:
            test_tensor = comm.recv(source=0)
            print(f"testing: {test_tensor.shape}")

        if comm_size > 4 and rank == 0: print(f"allgatherv: {input.shape}")
        output=comm.allgather(input.clone().detach())
        if comm_size > 4 and rank == 0: print(f"allgatherv: {output.shape}")
        output=torch.cat(output,dim_)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function._ContextMethodMixin, grad_output):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
        grad_format= torch.channels_last   if grad_output.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
        comm.barrier()
        output=comm.scatter(split_tensor_along_dim(grad_output, ctx.dim, comm_size))
        output = output.contiguous(memory_format=grad_format)
        return (output, None)

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
    return ScatterFunction.apply(input, dim)
    return _ScatterToParallelRegion.apply(input, dim)


def gather_from_parallel_region(input, dim, shapes):  # pragma: no cover
    """Gather the input from matmul parallel region and concatenate."""
    return AllgatherVFunction.apply(input, dim)
    return _GatherFromParallelRegion.apply(input, dim, shapes)
