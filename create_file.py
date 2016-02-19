import h5py
import numpy as np

f = h5py.File("testfile.h5", "w")

dset = f.create_dataset("mydataset", (10, 10, 10), dtype="f")
dset[:, :, :] = np.random.rand(10, 10, 10)
