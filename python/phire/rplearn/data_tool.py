from textwrap import indent
import h5py
import hdf5plugin
import dask.array as da
import numpy as np
import tensorflow as tf
import random


from ..utils import _int64_feature, _bytes_feature, _float_feature
from dask.diagnostics import ProgressBar


class DataSampler:

    def __init__(self):
        self.infile = '/data/ERA5/hdf5_hr/ds_train_1979_to_1990.hdf5'
        self.outfile = 'rplearn_train_1979_1990.{}.tfrecords'
        self.n_files = 8
        self.gzip = True
        self.shuffle = True

        ########################################################

        self.patch_size = (160,160)
        self.n_patches = 50
        self.T_max = 4 * 8 
        self.max_lat = 75

        self.log1p_norm = True
        self.z_norm = False
        
        self.mean, self.std = [2.0152406e-08, 2.1581373e-07], [2.8560082e-05, 5.0738556e-05]
        self.mean_log1p, self.std_log1p = [0.008315503, 0.0028762482], [0.5266841, 0.5418187]    

        ########################################################

        self.n_processed = 0
        self.N = None

    def calc_moments(self):
        with h5py.File(self.infile, 'r', rdcc_nbytes=1000*1000*1000) as f:
            data = da.from_array(f['data'], chunks=(-1, 256, -1, -1))  # CNHW layout
            data = da.transpose(data, (1,2,3,0))
            dtype = data.dtype

            if dtype != np.float32:
                print('WARNING: data will be saved as float32 but input ist float64!')

            if self.mean is None:
                arr = data
                with ProgressBar():
                    self.mean, self.std = da.compute(arr.mean(axis=[0,1,2]), arr.std(axis=[0,1,2]), num_workers=8)
            else:
                self.mean, self.std = np.asarray(self.mean, dtype=dtype), np.asarray(self.std, dtype=dtype)

            print('mean: {}, std: {}'.format(list(self.mean), list(self.std)))

            if self.log1p_norm:
                data_z_norm = (data-self.mean) / self.std
                data_log1p = da.sign(data_z_norm) * da.log1p(da.fabs(data_z_norm))

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
            log_mapped =  np.sign(z_normed) * np.log1p(np.fabs(z_normed))
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

            indices = np.random.permutation(self.N-self.T_max) if self.shuffle else np.arange###################(self.N-self.T_max)
            file_blocks = np.array_split(indices, self.n_files)

            for i, block in enumerate(file_blocks):
                if self.n_files > 1:
                    name = self.outfile.format(i)
                else:
                    name = self.outfile

                with tf.io.TFRecordWriter(name, options='ZLIB' if self.gzip else None) as writer:
                    self.write_records(writer, ds, block)
    

    def write_records(self, writer, ds, indices):
        n_warmup = 200
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
        for sample in samples:
            patch1, patch2, label, idx, (lat_start, long_start), (lat_end, long_end) = sample
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


    def add_to_queue(self, queue, ds, indices):
        C,_,H,W = ds.shape
        h_min = round((H / 180) * (90-self.max_lat))
        h_max = round((H / 180) * (260-self.max_lat)) - 1

        H_patch, W_patch = self.patch_size

        for idx in indices:
            img1 = self.fast_read(ds, idx)

            for _ in range(self.n_patches):
                label = + random.randint(1, self.T_max)
                lat,long = random.randint(h_min, h_max), random.randint(0, W+W_patch-1)

                # while this is a bit convoluted, it results in considerable speed gains due to the decreased file io
                lat_slice = slice(lat, lat+H_patch)
                long_slice = slice(long,long+W_patch)
                if long+self.patch_size[1] < W:
                    patch2 = self.fast_read(ds, idx + label, lat_slice, long_slice) 
                else:
                    img2 = self.fast_read(ds, idx + label, lat_slice) 
                    img2 = np.pad(img2, ((0,0),  (0, W_patch), (0,0)), 'wrap')
                    patch2 = img2[:, long_slice]

                patch1 = self.preprocess(img1[lat_slice, long_slice])
                patch2 = self.preprocess(patch2)

                start = (lat/H * 180, long/W * 360)
                end = ((lat+H_patch)/H * 180, (long+W_patch)/W * 360)

                queue.append((patch1, patch2, label, idx, start, end))

            self.n_processed += 1
            print('\r{:.2f}%'.format(100 * self.n_processed / self.N), flush=True, end='')


    def fast_read(self, ds, idx, h=slice(None), w=slice(None)):
        C = ds.shape[0]
        channels = tuple(ds[c, idx, h, w] for c in range(C))
        return np.stack(channels, axis=-1)


def parse_samples(serialized):
    feature = {
        'index': tf.FixedLenFeature([], tf.int64),
        'patch1': tf.FixedLenFeature([], tf.string),
        'patch2': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'T_max': tf.FixedLenFeature([], tf.int64),
        'H': tf.FixedLenFeature([], tf.int64),
        'W': tf.FixedLenFeature([], tf.int64),
        'C': tf.FixedLenFeature([], tf.int64),
        'lat_start': tf.FixedLenFeature([], tf.float64),
        'long_start': tf.FixedLenFeature([], tf.float64),
        'lat_end': tf.FixedLenFeature([], tf.float64),
        'long_end': tf.FixedLenFeature([], tf.float64)
    }
    
    examples = tf.parse_example(serialized, feature)
    return examples


def main():
    sampler = DataSampler()
    sampler.run()


if __name__ == '__main__':
    main()