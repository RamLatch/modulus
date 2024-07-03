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
    print(torch.equal(mpi, sgl))

tuples = [
    ("001_mpi_distributed_afnonet_forward","001_AFNO_x"),
    ("006_mpi_distributed_afnonet_patch_embed","002_PatchEmbed_x"),
    ("007_mpi_distributed_afnonet_pos_embed","005_AFNO_pos_embed"),
    ("008_mpi_distributed_afnonet_pos_drop","006_AFNO_pos_drop")

]

for t in tuples:
    compare(t)
