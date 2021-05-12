import tensorflow as tf
import h5py
import hdf5plugin
import dask.array as da
import numpy as np


# TF v1 Script!


class Downscaler:

    def __init__(self, sess, factor, img_shape):
        self.sess = sess
        self.factor = factor
        self.x_in = tf.placeholder(tf.float32, [None] + list(img_shape))
        self.pool = tf.nn.avg_pool2d(self.x_in, [1, factor, factor, 1], [1, factor, factor, 1],  padding='SAME')

    def downscale(self, x):
        return self.sess.run(self.pool, feed_dict={self.x_in: x})


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def patchify(data, y, x, patch_size):
    h,w = patch_size
    slices = []
    for i in range(x.shape[1]):
        for n in range(data.shape[0]):
            patch = data[n, y[n,i]:y[n,i]+h, x[n,i]:x[n,i]+w, ...]
            slices.append(patch)
    return slices


def patchify_random(data, patch_size, n_patches=14):
    h,w = patch_size

    y = np.random.randint(0, data.shape[1] - h, size=(data.shape[0], n_patches))
    x = np.random.randint(0, data.shape[2], size=(data.shape[0], n_patches))

    data = np.pad(data, ((0,0), (0,0), (0,w), (0,0)), 'wrap')
    patches = patchify(data, y, x, patch_size)
    return patches, y, x


def generate_TFRecords(writer, data, downscaler, mode, HR_patches):
    '''
    copied from utils.py and modified to allow appending
    '''    
    assert data.dtype == np.float32

    if mode == 'train':
        data_LR = downscaler.downscale(data)

    if HR_patches:
        data, y, x = patchify_random(data, HR_patches)
        if mode == 'train':
            K = downscaler.factor
            h,w = HR_patches[0] // K, HR_patches[1] // K
            data_LR = np.pad(data_LR, ((0,0), (0,0), (0,w), (0,0)), 'wrap')
            data_LR = patchify(data_LR, y // K, x // K, (h,w))

    for j in range(len(data)):  # works for both lists and np.ndarays
        if mode == 'train':
            h_HR, w_HR, c = data[j].shape
            h_LR, w_LR, c = data_LR[j].shape

            features = tf.train.Features(feature={
                                    'index': _int64_feature(j),
                                'data_LR': _bytes_feature(data_LR[j].tostring()),
                                    'h_LR': _int64_feature(h_LR),
                                    'w_LR': _int64_feature(w_LR),
                                'data_HR': _bytes_feature(data[j].tostring()),
                                    'h_HR': _int64_feature(h_HR),
                                    'w_HR': _int64_feature(w_HR),
                                        'c': _int64_feature(c)})
        elif mode == 'test':
            h_LR, w_LR, c = data[j].shape

            features = tf.train.Features(feature={
                                    'index': _int64_feature(j),
                                'data_LR': _bytes_feature(data[j].tostring()),
                                    'h_LR': _int64_feature(h_LR),
                                    'w_LR': _int64_feature(w_LR),
                                        'c': _int64_feature(c)})

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())


def main():
    ########################################################
    
    infile = '/data2/era5/ds_train_1980_1994.hdf5'
    outfile = '/data/stengel/patches/patches_train_1980_1994.{}.tfrecords'
    n_files = 8
    gzip = False
    shuffle = True

    ########################################################

    SR_ratio = 4
    log1p_norm = True
    z_norm = False
    
    mean, std = [275.28983, 1.8918675e-08, 2.3001131e-07], [16.951859, 2.19138e-05, 4.490682e-05]
    mean_log1p, std_log1p = [0.034322508, 0.01029128, 0.0031989873], [0.6344424, 0.53678083, 0.54819226]

    ################### #####################################

    HR_reduce_latitude = 32  # 11.25 deg from each pole
    HR_patches = (96, 96)

    ########################################################


    f = h5py.File(infile, 'r', rdcc_nbytes=1000*1000*1000)
    data = da.from_array(f['data'], chunks=(16 if shuffle else 256, -1, -1, -1))
    dtype = data.dtype

    if dtype != np.float32:
        print('WARNING: data will be saved as float32 but input ist float64!')

    if mean is None:
        mean, std = da.compute(data.mean(axis=[0,1,2]), data.std(axis=[0,1,2]))
    else:
        mean, std = np.asarray(mean, dtype=dtype), np.asarray(std, dtype=dtype)

    if log1p_norm:
        data_z_norm = (data-mean) / std
        data_log1p = da.sign(data_z_norm) * da.log1p(da.fabs(data_z_norm))

        if mean_log1p is None:
            mean_log1p, std_log1p = da.compute(data_log1p.mean(axis=[0,1,2]), data_log1p.std(axis=[0,1,2]))
        else:
            mean_log1p, std_log1p = np.asarray(mean_log1p, dtype=dtype), np.asarray(std_log1p, dtype=dtype)

        print('mean: {}, std: {}'.format(list(mean), list(std)))
        print('mean_log1p: {}, std_log1p: {}'.format(list(mean_log1p), list(std_log1p)))

        data = data_log1p
    elif z_norm:
        data = (data-mean) / std

    if not log1p_norm:
        print('mean: {}, std: {}'.format(list(mean), list(std)))

    if shuffle:
        block_indices = np.random.permutation(data.numblocks[0])
    else:
        block_indices = np.arange(data.numblocks[0])


    # Create session and start writing
    with tf.Session() as sess:
        if HR_reduce_latitude:
            H,W,C = data.shape[1:]
            downscaler = Downscaler(sess, SR_ratio, (H-HR_reduce_latitude, W, C))
        else:
            downscaler = Downscaler(sess, SR_ratio, data.shape[1:])

        file_blocks = np.array_split(block_indices, n_files)
        i = 0
        for n, indices in enumerate(file_blocks):
            if n_files > 1:
                name = outfile.format(n)
            else:
                name = outfile
            
            with tf.python_io.TFRecordWriter(name, options='ZLIB' if gzip else None) as writer:
                for block_idx in indices:
                    block =  data.blocks[block_idx].compute()
                    if shuffle:
                        block = np.random.permutation(block)

                    if HR_reduce_latitude:
                        lat_start = HR_reduce_latitude//2
                        block = block[:, lat_start:(-lat_start),:, :]

                    generate_TFRecords(writer, block, downscaler, 'train', HR_patches)
                    i += 1
                    print('{} / {}'.format(i, data.numblocks[0]))

if __name__ == '__main__':
    main()