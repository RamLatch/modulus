import pickle
import torch
PATH = "/hkfs/work/workspace/scratch/ie5012-MA/debug"

def compare(name1,name2):
    path1 = f"{PATH}/{name1}.pkl"
    path2 = f"{PATH}/{name2}.pkl"
    mpi = pickle.load(open(path1, "rb"))
    sgl = pickle.load(open(path2, "rb"))
    print(mpi)
    print(sgl)
    print(name1,name2,"\n",mpi.shape,sgl.shape)
    try:print(torch.allclose(mpi.type_as(sgl), sgl,rtol=0.0015))
    except: print(torch.eq(mpi.transpose(1,2),sgl))

tuples = [
    #("001_mpi_distributed_afnonet_Input","001_AFNO_Input"),
    #("002_mpi_distributed_patch_embed_Input","002_PatchEmbed_Input"),
    #("002_mpi_distributed_patch_embed_Input","002_mpi_distributed_patch_embed_copy_to_parallel_region"),
    ("002_mpi_distributed_patch_embed_return","002_PatchEmbed_return"),
    ("003_mpi_distributed_afnonet_patch_embed","003_AFNO_patch_embed"),
    ("004_mpi_distributed_afnonet_pos_embed","004_AFNO_pos_embed"),
    #("005_mpi_distributed_afnonet_pos_drop","005_AFNO_pos_drop"),
    #("005_mpi_distributed_afnonet_reshape","005_AFNO_reshape"),
    #("006_mpi_distributed_block_0_Input","006_Block_0_Input"),
    ("DistPatchembed_Conv2d_weight","Patchembed_Conv2d_weight"),
    ("DistPatchembed_Conv2d_bias","Patchembed_Conv2d_bias"),
    ("DistPatchembed_Conv2d_weight_after","Patchembed_Conv2d_weight_after"),
    ("DistPatchembed_Conv2d_bias_after","Patchembed_Conv2d_bias_after")]

for t in tuples:
    compare(t[0],t[1])
