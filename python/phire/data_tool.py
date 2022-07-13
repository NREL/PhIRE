from functools import WRAPPER_ASSIGNMENTS
from textwrap import indent
import h5py
import hdf5plugin
import dask.array as da
import numpy as np
import tensorflow as tf
import random


from phire.utils import _int64_feature, _bytes_feature, _float_feature
from dask.diagnostics import ProgressBar


class DataSampler:

    def __init__(self, mode, infile, outfile, patch_size, n_patches, T_max = None, lat_range=(0., 180.), long_range = (0., 360.), n_files=8, gzip=True, shuffle=True, K=4):
        self.mode = mode
        self.infile = infile
        self.outfile = outfile
        self.n_files = n_files
        self.gzip = gzip
        self.shuffle = shuffle

        ########################################################

        self.K = K
        self.patch_size = patch_size#(160,160)
        self.n_patches = n_patches#40
        self.T_max = T_max
        self.lat_range = lat_range
        self.long_range = long_range

        self.log1p_norm = True
        self.z_norm = False
        
        self.alpha = 0.2

        self.mean, self.std = [1.9464334e-08, 2.0547947e-07], [2.8568757e-05, 5.081943e-05]  # 1979-1988 div vort
        self.mean_log1p, self.std_log1p = [0.0008821452, 0.00032483143], [0.15794525, 0.16044095]  # alpha = 0.2
        
        ########################################################

        self.n_processed = 0
        self.N = None

        assert self.mode in ['rplearn', 'stengel-train', 'stengel-eval']
        assert self.patch_size[0] % 2 == 0
        assert self.patch_size[1] % 2 == 0


    def calc_moments(self):
        with h5py.File(self.infile, 'r', rdcc_nbytes=1000*1000*1000) as f:
            data = da.from_array(f['data'], chunks=(-1, 256, -1, -1))  # CNHW layout
            data = da.transpose(data, (1,2,3,0))
            dtype = data.dtype

            if dtype != np.float32:
                print('WARNING: data will be saved as float32 but input is float64!')

            if self.mean is None:
                arr = data
                with ProgressBar():
                    self.mean, self.std = da.compute(arr.mean(axis=[0,1,2]), arr.std(axis=[0,1,2]), num_workers=8)
            else:
                self.mean, self.std = np.asarray(self.mean, dtype=dtype), np.asarray(self.std, dtype=dtype)

            print('mean: {}, std: {}'.format(list(self.mean), list(self.std)))

            if self.log1p_norm:
                data_z_norm = (data-self.mean) / self.std
                data_log1p = da.sign(data_z_norm) * da.log1p(self.alpha * da.fabs(data_z_norm))

                if self.mean_log1p is None:
                    arr = data_log1p
                    with ProgressBar():
                        self.mean_log1p, self.std_log1p = da.compute(arr.mean(axis=[0,1,2]), arr.std(axis=[0,1,2]), num_workers=8)
                else:
                    self.mean_log1p, self.std_log1p = np.asarray(self.mean_log1p, dtype=dtype), np.asarray(self.std_log1p, dtype=dtype)

                print('mean_log1p: {}, std_log1p: {}'.format(list(self.mean_log1p), list(self.std_log1p)))


    def preprocess(self, arr):
        if self.log1p_norm:
            z_normed = (arr - self.mean) / self.std
            log_mapped =  np.sign(z_normed) * np.log1p(self.alpha * np.fabs(z_normed))
            return (log_mapped - self.mean_log1p) / self.std_log1p

        elif self.z_norm:
            return (arr - self.mean) / self.std

        else:
            return arr

    def run(self):
        self.calc_moments()

        with h5py.File(self.infile, 'r', rdcc_nbytes=1000*1000*1000) as f:
            ds = f['data']
            self.N = ds.shape[1] # CNHW layout!

            N = self.N - self.T_max if self.mode == 'rplearn' else self.N
            indices = np.random.permutation(N) if self.shuffle else np.arange(N)
            file_blocks = np.array_split(indices, self.n_files)

            for i, block in enumerate(file_blocks):
                if self.n_files > 1:
                    name = self.outfile.format(i)
                else:
                    name = self.outfile

                with tf.io.TFRecordWriter(name, options='ZLIB' if self.gzip else None) as writer:
                    self.write_records(writer, ds, block)
    

    def write_records(self, writer, ds, indices):
        n_warmup = 10000 if self.shuffle else 1
        queue = []

        self.add_to_queue(queue, ds, indices[:n_warmup])
        for idx in indices[n_warmup:]:
            self.add_to_queue(queue, ds, [idx])  # gurantees that this gets shuffled for remainder
            if self.shuffle:
                random.shuffle(queue)
            
            samples = queue[:self.n_patches]
            queue = queue[self.n_patches:]

            self.write(writer, samples)

        # write remaining samples (already shuffled)
        self.write(writer, queue)


    def write(self, writer, samples):
        if self.mode == 'rplearn':
            self.write_rplearn(writer, samples)
        else:
            self.write_stengel(writer, samples)


    def write_rplearn(self, writer, samples):
        for sample in samples:
            patch1, patch2, label, idx, (lat_start, long_start), (lat_end, long_end) = sample

            assert patch1.shape[:2] == self.patch_size
            assert patch2.shape[:2] == self.patch_size

            features = tf.train.Features(feature={
                'index': _int64_feature(idx),
                'patch1': _bytes_feature(patch1.tobytes()),
                'patch2': _bytes_feature(patch2.tobytes()),
                'label': _int64_feature(label),
                'T_max': _int64_feature(self.T_max),
                'H': _int64_feature(patch1.shape[0]),
                'W': _int64_feature(patch1.shape[1]),
                'C': _int64_feature(patch1.shape[2]),
                'lat_start': _float_feature(lat_start),
                'long_start': _float_feature(long_start),
                'lat_end': _float_feature(lat_end),
                'long_end': _float_feature(long_end)
            })
        
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

    
    def write_stengel(self, writer, samples):
        K = self.K

        for sample in samples:
            HR, idx, (lat_start, long_start), (lat_end, long_end) = sample

            assert HR.shape[:2] == self.patch_size

            if self.mode == 'stengel-train':
                # add batch dim and then remove it again
                LR = tf.nn.avg_pool2d(HR[None,:,:,:], [1, K, K, 1], [1, K, K, 1],  padding='SAME').numpy()[0]

                h_HR, w_HR, c = HR.shape
                h_LR, w_LR, c = LR.shape

                features = tf.train.Features(feature={
                                        'index': _int64_feature(idx),
                                    'data_LR': _bytes_feature(LR.tobytes()),
                                        'h_LR': _int64_feature(h_LR),
                                        'w_LR': _int64_feature(w_LR),
                                    'data_HR': _bytes_feature(HR.tobytes()),
                                        'h_HR': _int64_feature(h_HR),
                                        'w_HR': _int64_feature(w_HR),
                                            'c': _int64_feature(c)})
            elif self.mode == 'stengel-eval':
                h_LR, w_LR, c = HR.shape

                features = tf.train.Features(feature={
                                        'index': _int64_feature(idx),
                                    'data_LR': _bytes_feature(HR.tobytes()),
                                        'h_LR': _int64_feature(h_LR),
                                        'w_LR': _int64_feature(w_LR),
                                            'c': _int64_feature(c)})
            else:
                raise ValueError('invalid mode')

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


    def add_to_queue(self, queue, ds, indices):
        C,_,H,W = ds.shape

        # min and max y (latitude) in pixels
        h_min = round((H / 180) * self.lat_range[0])
        h_max = round((H / 180) * self.lat_range[1]) -1

        # min and max x (longitude) in pixels
        w_min = round((W / 360) * self.long_range[0])
        w_max = round((W / 360) * self.long_range[1]) -1

        H_patch, W_patch = self.patch_size

        # assert that patch does not exceed image boundaries        
        assert  H_patch//2 <= H-h_max

        for idx in indices:
            if self.mode == 'rplearn':
                img1 = self.fast_read(ds, idx)
                img1 = np.pad(img1, ((0,0),  (0, W_patch), (0,0)), 'wrap')

                for _ in range(self.n_patches):
                    label = random.randint(1, self.T_max)  # inclusive range
                    lat,long = random.randint(h_min, h_max), random.randint(w_min, w_max)

                    # while this is a bit convoluted, it results in considerable speed gains due to decreased file io
                    lat_slice = slice(lat-H_patch//2, lat+H_patch//2)
                    long_slice = slice(long,long+W_patch)
                    if long+W_patch < W:
                        patch2 = self.fast_read(ds, idx + label, lat_slice, long_slice) 
                    else:
                        img2 = self.fast_read(ds, idx + label, lat_slice) 
                        img2 = np.pad(img2, ((0,0),  (0, W_patch), (0,0)), 'wrap')
                        patch2 = img2[:, long_slice]

                    patch1 = self.preprocess(img1[lat_slice, long_slice])
                    patch2 = self.preprocess(patch2)

                    start = ((lat-H_patch//2)/H * 180, long/W * 360)
                    end = ((lat+H_patch//2)/H * 180, (long+W_patch)/W * 360)

                    queue.append((patch1, patch2, label, idx, start, end))
            
            else:
                if not self.n_patches:
                    img1 = self.fast_read(ds, idx, slice(h_min, h_max), slice(w_min, w_max))
                    HR = self.preprocess(img1)
                    start = (self.lat_range[0], self.long_range[0])
                    end = (self.lat_range[1], self.long_range[1])
                    queue.append((HR, idx, start, end))

                else:
                    img1 = self.fast_read(ds, idx)
                    img1 = np.pad(img1, ((0,0),  (0, W_patch), (0,0)), 'wrap')

                    for _ in range(self.n_patches):
                        lat,long = random.randint(h_min, h_max), random.randint(w_min, w_max)
                        lat_slice = slice(lat-H_patch//2, lat+H_patch//2)
                        long_slice = slice(long,long+W_patch)

                        HR = self.preprocess(img1[lat_slice, long_slice])

                        start = ((lat-H_patch//2)/H * 180, long/W * 360)
                        end = ((lat+H_patch//2)/H * 180, (long+W_patch)/W * 360)

                        queue.append((HR, idx, start, end))

            self.n_processed += 1
            print('\r{:.2f}%'.format(100 * self.n_processed / self.N), flush=True, end='')


    def fast_read(self, ds, idx, h=slice(None), w=slice(None)):
        C = ds.shape[0]
        channels = tuple(ds[c, idx, h, w] for c in range(C))
        return np.stack(channels, axis=-1)


rplearn_features = {
    'index': tf.io.FixedLenFeature([], tf.int64),
    'patch1': tf.io.FixedLenFeature([], tf.string),
    'patch2': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'T_max': tf.io.FixedLenFeature([], tf.int64),
    'H': tf.io.FixedLenFeature([], tf.int64),
    'W': tf.io.FixedLenFeature([], tf.int64),
    'C': tf.io.FixedLenFeature([], tf.int64),
    'lat_start': tf.io.FixedLenFeature([], tf.float32),
    'long_start': tf.io.FixedLenFeature([], tf.float32),
    'lat_end': tf.io.FixedLenFeature([], tf.float32),
    'long_end': tf.io.FixedLenFeature([], tf.float32)
}

def parse_samples(serialized):
    examples = tf.io.parse_example(serialized, rplearn_features)
    return examples


def rplearn_main():
    T_max = 4*8 - 1
    if True:
        sampler = DataSampler(
            'rplearn',
            infile = '/data/ERA5/hdf5_hr/ds_train_1979_1998.hdf5',
            outfile = 'rplearn_train_1979_1998.{}.hdf5',
            patch_size = (160, 160),
            n_patches = 20,
            T_max = T_max,
            lat_range=(30, 180 - 30),
            n_files = 2
        )
        sampler.run()
    else:
        sampler = DataSampler(
            'rplearn',
            infile = '/data/ERA5/hdf5_hr/ds_eval_2000_2005.hdf5',
            outfile = 'rplearn_eval_2000_2005.{}.hdf5',
            patch_size = (160, 160),
            n_patches = 20,
            T_max = T_max,
            lat_range=(30, 180 - 30),
            n_files = 2
        )
        sampler.run()


def phire_main():
    if True:
        sampler = DataSampler(
            'stengel-train',
            infile = '/data/ERA5/hdf5_hr/ds_train_1979_1998.hdf5',
            outfile = 'sr_train_1979_1998.{}.hdf5',
            patch_size = (96, 96),
            n_patches = 20,
            lat_range=(30, 180 - 30),
            n_files=2
        )
        sampler.run()

    else:
        sampler = DataSampler(
            'stengel-train',
            infile = '/data/ERA5/hdf5_hr/ds_eval_2000_2005.hdf5',
            outfile = 'sr_eval_2000_2005.{}.hdf5',
            patch_size = (-2, -2),  # unused
            n_patches = 0,  # generate full images,
            shuffle = False,
            n_files=2
        )
        sampler.run()