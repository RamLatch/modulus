import pickle
import torch
PATH = "/hkfs/work/workspace/scratch/ie5012-MA/debug"

def compare(name1,name2,name3=None):
    path1 = f"{PATH}/{name1}.pkl"
    path2 = f"{PATH}/{name2}.pkl"
    mpi = pickle.load(open(path1, "rb"))
    sgl = pickle.load(open(path2, "rb"))
    if name3 is not None:
        path3 = f"{PATH}/{name3}.pkl"
        sgl_imag = pickle.load(open(path3, "rb"))
        sgl = torch.stack([sgl[...,None],sgl_imag[...,None]],dim=-1)
    # print(mpi)
    # print(sgl)
    print(name1,name2,"\n",mpi.shape,sgl.shape)
    try:
        print(torch.allclose(mpi.type_as(sgl), sgl,rtol=0.0015))
        mpi = mpi.type_as(sgl)
        
    except: 
        try: 
            print(torch.allclose(mpi.type_as(sgl).transpose(-2,-1),sgl))
            mpi = mpi.type_as(sgl).transpose(-2,-1)
        except: 
            try: 
                print(torch.allclose(mpi.type_as(sgl).transpose(1,2).transpose(-2,-1),sgl))
                mpi = mpi.type_as(sgl).transpose(1,2).transpose(-2,-1)
            except: print(torch.allclose(mpi.type_as(sgl),sgl))
    print(mpi)
    print(sgl)

tuples = [
    #("001_mpi_distributed_afnonet_Input","001_AFNO_Input"),
    #("002_mpi_distributed_patch_embed_Input","002_PatchEmbed_Input"),
    #("002_mpi_distributed_patch_embed_Input","002_mpi_distributed_patch_embed_copy_to_parallel_region"),
    #("002_mpi_distributed_patch_embed_return","002_PatchEmbed_return"),
    #("003_mpi_distributed_afnonet_patch_embed","003_AFNO_patch_embed"),
    #("004_mpi_distributed_afnonet_pos_embed","004_AFNO_pos_embed"),
    #("005_mpi_distributed_afnonet_pos_drop","005_AFNO_pos_drop"),
    #("005_mpi_distributed_afnonet_reshape","005_AFNO_reshape"),
    ("006_mpi_distributed_block_0_Input","006_Block_0_Input"),
    ("006_mpi_distributed_block_0norm1","006_Block_0_norm1"),
    ("007_mpi_distributed_afno2d_view","007_AFNO2DLayer_x_real","007_AFNO2DLayer_x_imag"),
    #("DistPatchembed_Conv2d_weight","Patchembed_Conv2d_weight"),
    #("DistPatchembed_Conv2d_bias","Patchembed_Conv2d_bias"),
    #("DistPatchembed_Conv2d_weight_after","Patchembed_Conv2d_weight_after"),
    #("DistPatchembed_Conv2d_bias_after","Patchembed_Conv2d_bias_after")
    ]

for t in tuples:
    compare(t[0],t[1])
