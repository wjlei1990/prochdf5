from __future__ import print_function
import h5py
import numpy as np


class MatrixRun(object):

    def __init__(self, filename, dataset_name, output_dataset):

        self.filename = filename
        self.dataset_name = dataset_name
        self.output_dataset = output_dataset

        self.mpi_mode = False
        self.comm = None
        self.rank = None
        self._detect_env()

    def _detect_env(self):
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            if self.comm.size == 1:
                self.mpi_mode = False
                return
            self.rank = self.comm.rank
            self.mpi_mode = True
        except:
            self.mpi_mode = False

    @staticmethod
    def _core(data):
        return np.mean(data)

    def _job_partition(self, total_jobs):

        def split(container, count):
            """
            Simple function splitting a container into equal length chunks.
            Order is not preserved but this is potentially an advantage
            depending on the use case.
            """
            return [container[_i::count] for _i in range(count)]

        container = range(0, total_jobs)
        job_list = split(container, self.comm.size)
        return job_list[self.rank]

    def _load_hdf5(self, filename, permission="a"):
        if self.mpi_mode:
            return h5py.File(filename, permission, driver="mpio",
                             comm=self.comm)
        else:
            return h5py.File(filename, permission)

    def smart_run(self):

        f = self._load_hdf5(self.filename)

        if self.dataset_name not in f.keys():
            raise ValueError("dataset(%s) not in file(%s)"
                             % (self.dataset_name, f.keys()))

        dataset = f[self.dataset_name]

        try:
            shape = dataset.shape
        except:
            raise ValueError("It seems '%s' is not a dataset name"
                             % self.dataset_name)

        if len(shape) != 3:
            raise ValueError("Dimension of dataset: %s" % shape)

        if self.mpi_mode:
            job_list = self._job_partition(shape[0])
        else:
            job_list = range(shape[0])

        if self.mpi_mode:
            print("Rank %d joblist: %s" % (self.rank, job_list))

        final_result = np.zeros(shape[0:2])
        for i in job_list:
            if self.mpi_mode:
                print("rank, job:", self.rank, i)
            else:
                print("job:", i)
            for j in range(shape[1]):
                result = self._core(dataset[i, j])
            final_result[i, j] = result

        if self.output_dataset in f.keys():
            if f[self.output_dataset].shape != final_result.shape:
                raise ValueError("dataset(%s) already in file(%s)")
            else:
                f.__delitem__(self.output_dataset)
        od = f.create_dataset(self.output_dataset, final_result.shape,
                              dtype='f')
        od[:, :] = final_result

        f.close()
        if self.mpi_mode:
            print("Job Done on Rank %d" % self.rank)


if __name__ == "__main__":
    mr = MatrixRun("testfile.h5", "mydataset", "result")
    mr.smart_run()
