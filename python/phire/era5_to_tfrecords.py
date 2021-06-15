import tensorflow as tf
import h5py
import hdf5plugin
import dask.array as da
import numpy as np

from .utils import _bytes_feature, _int64_feature
from dask.diagnostics import ProgressBar


def patchify(data, y, x, patch_size):
    h,w = patch_size
    slices = []
    for i in range(x.shape[1]):
        for n in range(data.shape[0]):
            patch = data[n, y[n,i]:y[n,i]+h, x[n,i]:x[n,i]+w, ...]
            slices.append(patch)
    return slices


def patchify_random(data, patch_size, n_patches):
    h,w = patch_size

    y = np.random.randint(0, data.shape[1] - h, size=(data.shape[0], n_patches))
    x = np.random.randint(0, data.shape[2], size=(data.shape[0], n_patches))

    data = np.pad(data, ((0,0), (0,0), (0,w), (0,0)), 'wrap')
    patches = patchify(data, y, x, patch_size)
    return patches, y, x


def generate_TFRecords(writer, data, K, mode, HR_patches, n_patches):
    '''
    copied from utils.py and modified to allow appending
    '''    
    assert data.dtype == np.float32

    if mode == 'train':
        data_LR = tf.nn.avg_pool2d(data, [1, K, K, 1], [1, K, K, 1],  padding='SAME')

    if HR_patches:
        data, y, x = patchify_random(data, HR_patches, n_patches)
        if mode == 'train':
            h,w = HR_patches[0] // K, HR_patches[1] // K
            data_LR = np.pad(data_LR, ((0,0), (0,0), (0,w), (0,0)), 'wrap')
            data_LR = patchify(data_LR, y // K, x // K, (h,w))

    for j in range(len(data)):  # works for both lists and np.ndarays
        if mode == 'train':
            h_HR, w_HR, c = data[j].shape
            h_LR, w_LR, c = data_LR[j].shape

            features = tf.train.Features(feature={
                                    'index': _int64_feature(j),
                                'data_LR': _bytes_feature(data_LR[j].tobytes()),
                                    'h_LR': _int64_feature(h_LR),
                                    'w_LR': _int64_feature(w_LR),
                                'data_HR': _bytes_feature(data[j].tobytes()),
                                    'h_HR': _int64_feature(h_HR),
                                    'w_HR': _int64_feature(w_HR),
                                        'c': _int64_feature(c)})
        elif mode == 'test':
            h_LR, w_LR, c = data[j].shape

            features = tf.train.Features(feature={
                                    'index': _int64_feature(j),
                                'data_LR': _bytes_feature(data[j].tobytes()),
                                    'h_LR': _int64_feature(h_LR),
                                    'w_LR': _int64_feature(w_LR),
                                        'c': _int64_feature(c)})

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())


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

    #########################################################

    HR_reduce_latitude = 107  # 15 deg from each pole
    HR_patches = (96, 96)
    n_patches = 50

    ########################################################

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

    if shuffle:
        block_indices = np.random.permutation(data.numblocks[0])
    else:
        block_indices = np.arange(data.numblocks[0])


    file_blocks = np.array_split(block_indices, n_files)
    i = 0
    for n, indices in enumerate(file_blocks):
        if n_files > 1:
            name = outfile.format(n)
        else:
            name = outfile
        
        with tf.io.TFRecordWriter(name, options='ZLIB' if gzip else None) as writer:
            for block_idx in indices:
                block =  data.blocks[block_idx].compute()
                if shuffle:
                    block = np.random.permutation(block)

                if HR_reduce_latitude:
                    lat_start = HR_reduce_latitude//2
                    block = block[:, lat_start:(-lat_start),:, :]

                generate_TFRecords(writer, block, SR_ratio, 'train', HR_patches, n_patches)
                i += 1
                print('{} / {}'.format(i, data.numblocks[0]))

if __name__ == '__main__':
    main()
