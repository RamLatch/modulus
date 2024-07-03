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

# TODO this also needs more docstrings
from typing import List, Optional


from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

comm = MPI.COMM_WORLD

def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    # treat trivial case first
    if num_chunks == 1:
        return [size]

    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


def get_memory_format(tensor):
    """Gets format for tensor"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format


def pad_helper(tensor, dim, new_size, mode="zero"):
    """Util for padding tensors"""
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode="constant", value=0.0)

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad[lhs_slice] = torch.flip(
            torch.conj(tensor_pad[rhs_slice]), dims=[dim]
        )

    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    """Util for truncating"""
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[output_slice].contiguous(memory_format=input_format)

    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    if dim >= tensor.dim():
        raise ValueError(
            f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
        )
    if tensor.shape[dim] < num_chunks:
        raise ValueError(
            "Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
        {num_chunks} chunks. Empty slices are currently not supported."
        )

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)

    return tensor_list


@torch.no_grad()
def reduce_loss(loss: float, dst_rank: int = 0, mean: bool = True):  # pragma: no cover
    """Reduces loss from all processes to destination rank for logging.

    Parameters
    ----------
    loss : float
        loss value
    dst_rank : int, Optional
        destination rank to redce to, by default 0.
    mean : bool, Optional
        Calculate the mean of the losses gathered, by default True.

    Raises
    ------
    Exception
        If DistributedManager has yet to be initialized
    """
    if not MPI.Is_initialized():
        raise Exception(
            "MPI should be initialized when using reduce_loss"
        )
    rank = comm.Get_rank()
    size = comm.Get_size()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss = torch.Tensor([loss]).to(device)

    # For serial runs, just return the current loss!
    if size == 1:
        return float(loss)
    
    loss_sum = comm.reduce(loss, op=MPI.SUM, root=dst_rank)

    if rank == dst_rank:
        if mean:
            loss_sum /= size
        return loss_sum
    else:
        return None
    
    #tmp_loss = torch.zeros_like(loss)
    #comm.Reduce([loss, MPI.DOUBLE], tmp_loss, op=MPI.SUM, root=dst_rank)
    # Return loss if dst_rank, None otherwise
    #if mean and rank == dst_rank:
    #    return float(tmp_loss.cpu() / size)
    #elif rank == dst_rank:
    #    return float(tmp_loss.cpu())
    #else:
    #    return None


# distributed primitives
def distributed_transpose(tensor, dim0, dim1, async_op=False):
    """Perform distributed transpose of tensor to switch sharding dimension"""
    # get input format
    input_format = get_memory_format(tensor)

    # get comm params
    comm_size = comm.Get_size()

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [
        y.contiguous(memory_format=input_format)
        for y in torch.split(tensor, split_size, dim=dim0)
    ]
    x_recv = [torch.empty_like(x_send[0]) for _ in range(comm_size)]

    # global transposition
    req = comm.Alltoallv(x_send, x_recv)

    return x_recv, req

def _reduce_torch(input_, use_fp32=True):
    """All-reduce the input tensor across model parallel group using torch.distributed."""
    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size() == 1:
        return input_
    
    # All-reduce, use_fp32 only relevant for lower precisions
    if use_fp32 and (input_.dtype.itemsize < 4) and input_.dtype.is_floating_point:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, op=dist.ReduceOp.SUM)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, op=dist.ReduceOp.SUM)

    return input_

def _reduce(input_, use_fp32=True):  # pragma: no cover
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if comm.Get_size() == 1:
        return input_
    comm.Allreduce(MPI.IN_PLACE, input_, op=MPI.SUM)
    return input_
    # All-reduce, use_fp32 only relevant for lower precisions
    # if input is already in double precision, nothing changes
    # if use_fp32 and (input_.dtype.itemsize < 4) and input_.dtype.is_floating_point:
    #     dtype = input_.dtype
    #     inputf_ = input_.float()
    #     comm.Allreduce(inputf_, inputf_, op=MPI.SUM)
    #     input_ = inputf_.to(dtype)
    # else:
    #     comm.Allreduce(input_, input_, op=MPI.SUM)

    # return input_


def _split(input_, dim_):  # pragma: no cover
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = comm.Get_size()
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = comm.Get_rank()
    output = input_list[rank].contiguous(memory_format=input_format)

    return output



def all_gather_v_wrapper_torch(
        tensor: torch.Tensor,
        sizes: Optional[List[int]] = None,
        dim: int = 0
    ) -> torch.Tensor:
        """
        Implements a distributed AllGatherV primitive using torch.distributed.
        It gathers all local tensors from each rank into the full global tensor onto each rank.

        Parameters
        ----------
        tensor : torch.Tensor
            Local tensor on each rank
        sizes : List[int], optional
            List of the sizes of each chunk on each rank along distributed dimension,
            valid and set on each rank, by default None
        dim : int, optional
            Dimension along which global tensor is distributed, by default 0

        Returns
        -------
        torch.Tensor
            Full global tensor, valid on each rank
        """
        world_size = dist.get_world_size()

        if (sizes is not None) and (len(sizes) != world_size):
            raise ValueError("Sizes list length must be equal to the world size")

        if dim >= tensor.dim():
            raise ValueError("Invalid dimension")

        if world_size == 1:
            return tensor.clone()

        # Determine local tensor size
        local_size = tensor.size(dim)
        local_sizes = [torch.tensor(local_size)] * world_size

        if sizes is None:
            sizes = local_sizes

        # Calculate total size for the receive buffer and displacements
        total_size = sum(sizes)
        displacements = [sum(sizes[:i]) for i in range(world_size)]

        # Prepare the receive buffer
        recv_buf = torch.empty(total_size, dtype=tensor.dtype, device=tensor.device)

        # Flatten the tensor for sending
        send_data = tensor.flatten()

        # Perform AllGatherV operation
        dist.all_gather(tensor_list=[recv_buf], tensor=send_data, sizes=sizes, dim=dim)

        # Reconstruct the global tensor
        global_tensor = recv_buf.view(-1, *tensor.size()[1:]).to(tensor.device)
        global_tensor = global_tensor.type(tensor.dtype)

        return global_tensor

def all_gather_v_wrapper(
    tensor: torch.Tensor,
    sizes: Optional[List[int]] = None,
    dim: int = 0
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed AllGatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int], optional
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank, by default None
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """

    comm_size = comm.Get_size()

    if (sizes is not None) and (len(sizes) != comm_size):
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    if comm_size == 1:
        return tensor

    # Determine local tensor size
    local_size = tensor.size(dim)
    local_sizes = comm.allgather(local_size)  # Gather sizes of tensors from all ranks

    if sizes is None:
        sizes = local_sizes

    # Calculate total size for the receive buffer and displacements
    total_size = sum(sizes)
    displacements = [sum(sizes[:i]) for i in range(comm_size)]

    # Prepare the receive buffer
    recv_buf = np.empty(total_size, dtype=tensor.numpy().dtype)

    # Flatten the tensor for sending
    send_data = tensor.numpy().flatten()

    # Perform Allgatherv operation
    comm.Allgatherv(send_data, [recv_buf, sizes, displacements, MPI.DOUBLE])

    # Reconstruct the global tensor
    # Assuming the tensor is 1D for simplicity. Adjust for actual dimensions.
    global_tensor = torch.from_numpy(recv_buf).view(-1, *tensor.size()[1:]).to(tensor.device)
    global_tensor=global_tensor.type(tensor.dtype)
    return global_tensor
    # tensor_shape = list(tensor.shape)
    # tensor_format = get_memory_format(tensor)

    # if sizes is not None:
    #     tensor_list = [None] * comm_size

    #     for src in range(comm_size):
    #         tensor_shape[dim] = sizes[src]
    #         tensor_list[src] = torch.empty(
    #             tensor_shape,
    #             dtype=tensor.dtype,
    #             device=tensor.device,
    #         )
    # else:
    #     # assume equal shape on all ranks
    #     tensor_list = [torch.empty_like(tensor) for _ in range(comm_size)]

    # comm.Allgatherv(tensor, tensor_list)

    # output = torch.cat(tensor_list, dim=dim).contiguous(memory_format=tensor_format)

    # return output


def all_gather_v_bwd_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    use_fp32: bool = True
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed AllReduceV primitive. It is based
    on the idea of a single global tensor which which can be distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes different global tensors of the same shape on each
    rank. It then re-distributes chunks of all these tensors such that each rank
    receives all corresponding parts of a global tensor. Each rank then sums up
    the chunks after receiving it. By design, this primitive thus implements the
    backward pass of the "all_gather_v" primitive. In this case, the result would
    be a single global gradient tensor distributed onto different ranks.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor on each rank (different one on each rank)
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local tensor, i.e. result of reduction of all corresponding chunks
        from all global tensors for each rank separately
    """

    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    # Convert PyTorch tensor to NumPy
    if use_fp32:
        tensor_np = tensor.numpy().astype(np.float32)
    else:
        tensor_np = tensor.numpy()

    # Gather sizes from all ranks
    recv_sizes = np.zeros(comm_size, dtype=int)
    comm.Allgather(np.array([tensor_np.size], dtype=int), recv_sizes)

    # Calculate displacements for each rank
    displs = np.zeros(comm_size, dtype=int)
    displs[1:] = np.cumsum(recv_sizes[:-1])

    # Prepare receive buffer
    recvbuf = np.empty(sum(recv_sizes), dtype=tensor_np.dtype)

    # Allgather variable-sized chunks
    comm.Allgatherv(sendbuf=tensor_np, recvbuf=(recvbuf, recv_sizes, displs), root=0)

    # Convert back to PyTorch tensor and sum up the chunks
    gathered_tensor = torch.from_numpy(recvbuf)
    output = torch.sum(gathered_tensor, dim=dim)

    return output
    # tensor_shape = list(tensor.shape)
    # tensor_shape[dim] = sizes[rank]
    # tmp = [
    #     torch.empty(
    #         tensor_shape,
    #         dtype=tensor.dtype,
    #         device=tensor.device,
    #     )
    #     for _ in range(comm_size)
    # ]
    # scatter_list = list(torch.split(tensor, sizes, dim=dim))
    # scatter_list = [t.contiguous() for t in scatter_list]

    # comm.Alltoallv(scatter_list, tmp)
    # stack_dim = tensor.dim()
    # tmp = torch.stack(tmp, dim=stack_dim)

    # if use_fp32 and (tmp.dtype.itemsize < 4) and tmp.dtype.is_floating_point:
    #     # cast to float before sum and return float, then cast back
    #     output = tmp.sum(dim=stack_dim, dtype=torch.float32)
    #     output = output.to(dtype=tensor.dtype)
    # else:
    #     # else: just do sum in native dtype
    #     output = tmp.sum(dim=stack_dim)

    # return output


def gather_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    dst: int = 0
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed GatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    dst : int, optional
        destination rank which contains the full global tensor after the
        operation, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on destination rank
    """

    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if not (0 <= dst < comm_size):
        raise ValueError()
    if tensor.size(dim) != sizes[rank]:
        raise ValueError()

    if comm_size == 1:
        return tensor
    # Ensure tensor is a numpy array for MPI operations
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()

    # Transpose tensor if necessary
    if dim != 0:
        tensor = tensor.transpose((dim,) + tuple(range(dim)) + tuple(range(dim+1, tensor.ndim)))

    # Prepare sizes and displacements for Gatherv
    sizes = np.array(sizes) * tensor.itemsize
    displacements = np.insert(np.cumsum(sizes), 0, 0)[:-1]

    # Only the root process prepares the receive buffer
    if rank == dst:
        recvbuf = np.empty(sum(sizes) // tensor.itemsize, dtype=tensor.dtype)
    else:
        recvbuf = None

    # Perform the Gatherv operation
    comm.Gatherv(sendbuf=tensor, recvbuf=(recvbuf, sizes, displacements), root=dst)

    # Convert back to torch.Tensor if necessary
    if recvbuf is not None:
        recvbuf = torch.from_numpy(recvbuf)

    # Transpose back if necessary
    if dim != 0 and recvbuf is not None:
        recvbuf = recvbuf.transpose(0, dim)

    return recvbuf if rank == dst else None
    # tensor_shape = list(tensor.shape)
    # x_recv = [None] * comm_size
    # x_send = [None] * comm_size

    # for r in range(comm_size):
    #     if rank == dst:
    #         tensor_shape[dim] = sizes[r]
    #     else:
    #         tensor_shape[dim] = 0

    #     x_recv[r] = torch.empty(
    #         tensor_shape,
    #         dtype=tensor.dtype,
    #         device=tensor.device,
    #     )

    #     if r == dst:
    #         x_send[r] = tensor
    #     else:
    #         tensor_shape[dim] = 0
    #         x_send[r] = torch.empty(
    #             tensor_shape,
    #             dtype=tensor.dtype,
    #             device=tensor.device,
    #         )
    # comm.Gatherv(x_send[rank], x_recv, root=dst)

    # # TODO: clean gather/scatter and some examples up
    # # main question is around whether e.g. gather returns
    # # None for rank != dst or an empty dummy or an dummy
    # # containing meta-information like dtype/etc..
    # if rank != dst:
    #     for r in range(comm_size):
    #         tensor_shape[dim] = sizes[r]
    #         x_recv[r] = torch.empty(
    #             tensor_shape,
    #             dtype=tensor.dtype,
    #             device=tensor.device,
    #         )

    # output = torch.cat(x_recv, dim=dim)

    # return output


def scatter_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    src: int = 0
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed ScatterV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor, valid on source rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    src : int, optional
        source rank of primitive, i.e. rank of original full global tensor, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        corresponding local part of the global tensor on each rank
    """
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if len(sizes) != comm_size:
        raise ValueError()
    if comm.Get_rank() == 0 and dim >= tensor.dim():
        raise ValueError()
    if not (0 <= src < comm_size):
        raise ValueError()

    if rank == src:
        # Split the tensor using torch.split
        chunks = torch.split(tensor, sizes, dim=dim)
        # Convert each chunk to a NumPy array
        np_chunks = [chunk.numpy() for chunk in chunks]
        # Flatten and prepare the data for scattering
        sendbuf = np.concatenate([chunk.flatten() for chunk in np_chunks])
        sendcounts = [chunk.size for chunk in np_chunks]
        displs = [sum(sendcounts[:i]) for i in range(comm_size)]
    else:
        sendbuf = None
        sendcounts = None
        displs = None

    # Scatter the sizes first
    recvcount = comm.scatter(sizes, root=src)
    # Prepare a buffer to receive the scattered data
    recvbuf = np.empty(recvcount, dtype=np.float32)  # Adjust dtype as necessary

    # Scatter the data
    comm.Scatterv([sendbuf, sendcounts, displs, MPI.FLOAT], recvbuf, root=src)

    # Convert the received numpy array back to a PyTorch tensor
    scattered_tensor = torch.from_numpy(recvbuf)

    return scattered_tensor
    # all_to_all is already all_to_all_v, use empty tensors to "mask"-out irrelevant parts
    # tensor_shape = list(tensor.shape)
    # x_send = [None] * comm_size
    # x_recv = [None] * comm_size
    # if rank == src:
    #     scatter_list = torch.split(tensor, sizes, dim=dim)
    #     scatter_list = [t.contiguous() for t in scatter_list]
    #     x_send = scatter_list
    # else:
    #     for r in range(comm_size):
    #         tensor_shape[dim] = 0
    #         x_send[r] = torch.empty(
    #             tensor_shape, device=tensor.device, dtype=tensor.dtype
    #         )

    # for r in range(comm_size):
    #     if r == src:
    #         tensor_shape[dim] = sizes[rank]
    #     else:
    #         tensor_shape[dim] = 0
    #     x_recv[r] = torch.empty(tensor_shape, device=tensor.device, dtype=tensor.dtype)

    # comm.Scatterv([x_send, sizes, None, MPI.DOUBLE], x_recv, root=src)

    # return x_recv[src]


def indexed_all_to_all_v_wrapper(
    tensor: torch.Tensor,
    indices: List[torch.Tensor],
    sizes: List[List[int]],
    dim: int = 0
) -> torch.Tensor:  # pragma: no cover
    """
    Implements an indexed version of a distributed AllToAllV
    primitive. It is based on the idea of a single global tensor which
    is distributed along a specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local part of global tensor on each rank
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local result of primitive corresponding to indexed global tensor
    """

    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    # Flatten the tensor according to indices for sending
    send_data = torch.cat([tensor.index_select(dim, ind) for ind in indices], dim=dim).numpy()
    send_counts = [len(ind) for ind in indices]  # Number of elements to send to each rank
    sdispls = np.insert(np.cumsum(send_counts), 0, 0)[:-1]  # Displacements for sending

    # Prepare receive counts and displacements
    recv_counts = [sizes[i][rank] for i in range(comm_size)]
    rdispls = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]  # Displacements for receiving

    # Prepare receive buffer
    total_recv_elements = sum(recv_counts)
    recv_data = np.empty(total_recv_elements, dtype=send_data.dtype)

    # Perform Alltoallv operation
    comm.Alltoallv([send_data, send_counts, sdispls, MPI.DOUBLE],
                   [recv_data, recv_counts, rdispls, MPI.DOUBLE])

    # Reconstruct the received tensor
    # Assuming the received data is 1D for simplicity. Adjust for actual dimensions.
    recv_tensor = torch.from_numpy(recv_data).to(tensor.device)

    return recv_tensor
    # x_send = [tensor[idx] for idx in indices]
    # x_recv = [None] * comm_size
    # tensor_shape = list(tensor.shape)
    # for r in range(comm_size):
    #     tensor_shape[dim] = sizes[r][rank]
    #     x_recv[r] = torch.empty(
    #         tensor_shape,
    #         dtype=tensor.dtype,
    #         device=tensor.device,
    #     )

    # sendbuf = [x_send[r].numpy() for r in range(comm_size)]
    # recvbuf = [x_recv[r].numpy() for r in range(comm_size)]

    # comm.Alltoallv(sendbuf, sizes[rank], recvbuf)

    # tensor_to_recv = torch.cat([torch.from_numpy(recvbuf[r]) for r in range(comm_size)], dim=dim)

    # return tensor_to_recv


def indexed_all_to_all_v_wrapper_bwd(
    tensor: torch.Tensor,
    indices: List[torch.Tensor],
    sizes: List[List[int]],
    tensor_size_along_dim: int,
    use_fp32: bool = True,
    dim: int = 0
) -> torch.Tensor:  # pragma: no cover
    """
    Implements the backward pass to the indexed version of a distributed
    AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor, i.e. gradient on resulting tensor from forward pass
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    tensor_size_along_dim : int
        size of original local tensor along specified dimension,
        i.e. from the corresponding forward pass
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    dim : int, optional
        dimension along with global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        result of primitive corresponding to indexed global tensor
    """

    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    # Convert tensor to appropriate dtype
    if use_fp32 and tensor.dtype != torch.float32:
        tensor = tensor.float()

    # Flatten the tensor according to indices for sending
    send_data = torch.cat([tensor.index_select(dim, ind) for ind in indices], dim=dim).numpy()
    send_counts = [len(ind) for ind in indices]  # Number of elements to send to each rank
    sdispls = np.insert(np.cumsum(send_counts), 0, 0)[:-1]  # Displacements for sending

    # Prepare receive counts and displacements based on sizes from the forward pass
    recv_counts = [sizes[i][rank] for i in range(comm_size)]
    rdispls = np.insert(np.cumsum(recv_counts), 0, 0)[:-1]  # Displacements for receiving

    # Prepare receive buffer
    total_recv_elements = sum(recv_counts)
    recv_data = np.empty(total_recv_elements, dtype=send_data.dtype)

    # Perform Alltoallv operation
    comm.Alltoallv([send_data, send_counts, sdispls, MPI.DOUBLE],
                   [recv_data, recv_counts, rdispls, MPI.DOUBLE])

    # Reconstruct the received gradient tensor
    # Assuming the received data is 1D for simplicity. Adjust for actual dimensions.
    recv_tensor = torch.from_numpy(recv_data).to(tensor.device)

    # Convert back to original dtype if necessary
    if use_fp32 and tensor.dtype != torch.float32:
        recv_tensor = recv_tensor.to(tensor.dtype)

    return recv_tensor
    # tensor_shape = list(tensor.shape)

    # # scatter gradients, roles reversed compared to forward pass
    # # recv_sizes in forward pass
    # recv_sizes = [sizes[i][rank] for i in range(comm_size)]
    # # send_sizes in forward pass
    # send_sizes = [sizes[rank][i] for i in range(comm_size)]

    # x_send = [None] * comm_size
    # x_recv = [None] * comm_size

    # for r in range(comm_size):
    #     if rank == r:
    #         x_send[r] = tensor
    #     else:
    #         x_send[r] = torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)

    # comm.Scatterv([x_send, send_sizes, None, MPI.DOUBLE], x_recv, root=rank)

    # tensor_to_recv = torch.cat(x_recv, dim=dim)

    # # sum up gathered gradients and taking
    # # care of precision handling as specified
    # # by boolean flag
    # indices = torch.cat(indices, dim=0)
    # tensor_shape[dim] = tensor_size_along_dim
    # if use_fp32 and (tensor.dtype.itemsize < 4) and tensor.dtype.is_floating_point:
    #     out = torch.zeros(tensor_shape, dtype=torch.float32, device=tensor.device)
    #     tensor_to_recv = tensor_to_recv.to(dtype=torch.float32)
    # else:
    #     out = torch.zeros(tensor_shape, dtype=tensor.dtype, device=tensor.device)

    # out.index_add_(source=tensor_to_recv, index=indices, dim=dim)

    # if out.dtype != tensor.dtype:
    #     out = out.to(tensor.dtype)

    # return out


def mark_module_as_shared(
    module: nn.Module,
    process_group: Optional[str],
    recurse: bool = True,
    use_fp32_reduction: bool = True,
) -> nn.Module:
    """
    Helper function to mark parameters of a module as being shared
    across ranks by attaching gradient hooks to the corresponding tensors.

    Parameters
    ----------
    module : nn.Module
        PyTorch module which is to be marked as having shared parameters.
    process_group : str | None
        str indicating process_group which contains ranks across which
        the module's parameters are shared. If passed as None, will default
        to the world group.
    recurse : bool, default=True
        Flag indicating whether the module's parameters are traversed in
        a recursive fashion, i.e. whether sub-modules are also considered
        as having shared parameters.
    use_fp32_reduction : bool, default=True
        Flag indicating whether the reduction for accumulating gradients
        will be done in at least FP32 or the native datatype.
    """

    
    handle_key = "_shared_weight_dist_hook"

    def hook(grad: torch.Tensor) -> torch.Tensor:
        # the documentation states that
        # "The hook should not modify its argument, but it can optionally return a new gradient
        #  which will be used in place of grad."
        # as all_reduce is an in-place operation, need to copy gradient
        grad = _reduce(grad.clone(), use_fp32=use_fp32_reduction)
        return grad

    def hook_post_accum(param: torch.Tensor) -> None:
        # the documentation states that
        # "Note that, unlike other autograd hooks, this hook operates on the tensor that requires grad
        #  and not the grad itself. The hook can in-place modify and access its Tensor argument,
        # including its .grad field."
        param.grad = _reduce(param.grad, use_fp32=use_fp32_reduction)

    for name, param in module.named_parameters(recurse=recurse):
        error_msg = f"Parameter {name} already marked as having shared weights, can't mark it again!"
        if hasattr(param, handle_key):
            raise RuntimeError(error_msg)
        if torch.__version__ < (2, 1):
            handle = param.register_hook(hook)
        else:
            handle = param.register_post_accumulate_grad_hook(hook_post_accum)
        setattr(param, handle_key, handle)

    return module


def unmark_module_as_shared(
    module: nn.Module,
    recurse: bool = True,
) -> nn.Module:
    """
    Helper function to unmark parameters of a module as being shared
    across ranks by removing attached gradient hooks.

    Parameters
    ----------
    module : nn.Module
        PyTorch module which is to be unmarked as having shared parameters.
    recurse : bool, default=True
        Flag indicating whether the module's parameters are traversed in
        a recursive fashion, i.e. whether sub-modules are also considered
        as having shared parameters.
    """
    handle_key = "_shared_weight_dist_hook"
    for name, param in module.named_parameters(recurse=recurse):
        error_msg = (
            f"Parameter {name} NOT marked as having shared weights, can't unmark it!"
        )
        if not hasattr(param, handle_key):
            raise RuntimeError(error_msg)
        handle = getattr(param, handle_key)
        handle.remove()
        delattr(param, handle_key)

    return module
