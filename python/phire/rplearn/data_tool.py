import h5py
import dask.array as da
import numpy as np
import tensorflow as tf

from ..utils import _int64_feature, _bytes_feature
from dask.diagnostics import ProgressBar

def main():
    ########################################################
    
    infile = '/data/ERA5/hdf5_hr/ds_train_1979_to_1990.hdf5'
    outfile = 'patches_train_1979_1990.{}.tfrecords'
    n_files = 8
    gzip = True
    shuffle = True

    ########################################################

    SR_ratio = 4
    log1p_norm = True
    z_norm = False
    
    mean, std = [2.0152406e-08, 2.1581373e-07], [2.8560082e-05, 5.0738556e-05]
    mean_log1p, std_log1p = [0.008315503, 0.0028762482], [0.5266841, 0.5418187]

    f = h5py.File(infile, 'r', rdcc_nbytes=1000*1000*1000)
    data = da.from_array(f['data'], chunks=(-1, 16 if shuffle else 256, -1, -1))  # CNHW layout
    data = da.transpose(data, (1,2,3,0))
    dtype = data.dtype

    if dtype != np.float32:
        print('WARNING: data will be saved as float32 but input ist float64!')

    if mean is None:
        arr = data
        with ProgressBar():
            mean, std = da.compute(arr.mean(axis=[0,1,2]), arr.std(axis=[0,1,2]), num_workers=8)
    else:
        mean, std = np.asarray(mean, dtype=dtype), np.asarray(std, dtype=dtype)

    print('mean: {}, std: {}'.format(list(mean), list(std)))

    if log1p_norm:
        data_z_norm = (data-mean) / std
        data_log1p = da.sign(data_z_norm) * da.log1p(da.fabs(data_z_norm))

        if mean_log1p is None:
            arr = data_log1p
            with ProgressBar():
                mean_log1p, std_log1p = da.compute(arr.mean(axis=[0,1,2]), arr.std(axis=[0,1,2]), num_workers=8)
        else:
            mean_log1p, std_log1p = np.asarray(mean_log1p, dtype=dtype), np.asarray(std_log1p, dtype=dtype)

        print('mean_log1p: {}, std_log1p: {}'.format(list(mean_log1p), list(std_log1p)))

        data = data_log1p
    elif z_norm:
        data = (data-mean) / std

if __name__ == '__main__':
    main()