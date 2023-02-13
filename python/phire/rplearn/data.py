import tensorflow as tf
import numpy as np
from ..data_tool import parse_samples, rplearn_features


"""
def sin_cos_field(start, end, spatial_shape, axis):
    assert(axis == 1 or axis == 2)
    x  = tf.linspace(start, end, num=spatial_shape[axis], axis=-1)  # B x K (K=shape[axis])

    lat_sin = tf.math.sin(2*np.pi * x)
    lat_cos = tf.math.cos(2*np.pi * x)
    fields = tf.stack([lat_sin, lat_cos], axis=-1) # B x K x C
    
    fields = tf.expand_dims(fields, axis = 2 if axis == 1 else 1)
    fields = tf.broadcast_to(fields, spatial_shape + [2])

    return fields
"""


def parse_rplearn(serialized):
    examples = tf.io.parse_example(serialized, rplearn_features)
    
    N = tf.cast(tf.shape(examples['H'])[0], tf.int64)
    H,W,C = examples['H'][0], examples['W'][0], examples['C'][0]
    
    patch1 = tf.io.decode_raw(examples['patch1'], tf.float32)
    patch1 = tf.reshape(patch1, (-1, H,W,C))

    patch2 = tf.io.decode_raw(examples['patch2'], tf.float32)
    patch2 = tf.reshape(patch2, (-1, H,W,C))

    T_max = examples['T_max'][0]
    labels = tf.one_hot(examples['label'] - 1, tf.cast(T_max, tf.int32))

    return patch1, patch2, labels


def parse_atmodist_sample(serialized):
    patch1, patch2, labels = parse_rplearn(serialized)

    X = {'img1': patch1, 'img2': patch2}
    y = labels

    return X, y


def parse_autoencoder_sample(serialized):
    patch1, patch2, labels = parse_rplearn(serialized)

    imgs = tf.concat([patch1, patch2], axis=0)

    X = imgs
    y = imgs

    return X, y


def parse_inpaint_sample(serialized):
    X,y = parse_autoencoder_sample(serialized)

    s = tf.shape(X)
    N,H,W,C = s[0], s[1], s[2], s[3]

    H_l = 3*(H//8) - 1
    H_r = 5*(H//8) - 1
    W_l = 3*(W//8) - 1
    W_r = 5*(H//8) - 1

    # fill area with zeros
    indices = tf.range(N*H*W*C, dtype=tf.int32)
    W_indices = (indices // C) % W
    H_indices = (indices // (W*C)) % H
    mask = (H_indices >= H_l) & (H_indices < H_r) & (W_indices >= W_l) & (W_indices < W_r)
    X = tf.reshape(X, [-1])
    X = tf.where(mask, tf.zeros([N*H*W*C]), X)
    X = tf.reshape(X, [N,H,W,C])

    return X, y


def make_atmodist_ds(files, batch_size, T_max=None, n_shuffle=1000, compression_type='ZLIB'):
    assert files

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=4 if n_shuffle else None, compression_type=compression_type)
    
    if n_shuffle:
        ds = ds.shuffle(n_shuffle)

    ds = ds.batch(1)
    ds = ds.map(parse_atmodist_sample)
    ds = ds.unbatch()
    
    if T_max:
        ds = ds.filter(lambda X, y: tf.argmax(y) < T_max)
    
    ds = ds.batch(batch_size)
    return ds.prefetch(None)


def make_autoencoder_ds(files, batch_size, n_shuffle=1000, compression_type='ZLIB'):
    assert files

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=4 if n_shuffle else None, compression_type=compression_type)
    
    if n_shuffle:
        ds = ds.shuffle(n_shuffle)

    ds = ds.batch(1)
    ds = ds.map(parse_autoencoder_sample)
    ds = ds.unbatch()
    
    ds = ds.batch(batch_size)
    return ds.prefetch(None)


def make_inpaint_ds(files, batch_size, n_shuffle=1000, compression_type='ZLIB'):
    assert files

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=4 if n_shuffle else None, compression_type=compression_type)
    
    if n_shuffle:
        ds = ds.shuffle(n_shuffle)

    ds = ds.batch(1)
    ds = ds.map(parse_inpaint_sample)
    ds = ds.unbatch()
    
    ds = ds.batch(batch_size)
    return ds.prefetch(None)