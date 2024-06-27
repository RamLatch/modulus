import h5py
import numpy as np

# Open the original HDF5 file
original_file = h5py.File('/hkfs/work/workspace/scratch/ie5012-MA/data/train/1979.h5', 'r')

# Create a new HDF5 file for the copy
copy_file = h5py.File('/hkfs/work/workspace/scratch/ie5012-MA/data/testing/Dummy.h5', 'w')

# Iterate over all datasets in the original file
for dataset_name, dataset in original_file.items():
    # Read the dataset values
    dataset_values = dataset[()]

    # Create a new dataset in the copy file with the same shape and data type
    copy_dataset = copy_file.create_dataset(dataset_name, shape=dataset_values.shape, dtype=dataset_values.dtype)

    # Replace all values in the copy dataset with 1
    copy_dataset[()] = np.ones_like(dataset_values)

# Close the files
original_file.close()
copy_file.close()