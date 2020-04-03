import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def conv_layer_2d(x, filter_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    x = tf.nn.conv2d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, 1],
        padding='SAME')

    return x

def deconv_layer_2d(x, filter_shape, output_shape, stride, trainable=True):
    x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    x = tf.nn.conv2d_transpose(
        value=x,
        filter=filter_,
        output_shape=output_shape,
        strides=[1, stride, stride, 1])

    return x[:, 3:-3, 3:-3, :]

def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    x = tf.reshape(transposed, [-1, dim])

    return x

def dense_layer(x, out_dim, trainable=True):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.02),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    x = tf.add(tf.matmul(x, W), b)

    return x

def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        N, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, (N, h, w, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, h, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, w, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.reshape(x, (N, h*r, w*r, 1))

    xc = tf.split(x, n_split, 3)
    x = tf.concat([PS(x_, r) for x_ in xc], 3)

    return x

def plot_SR_data(idx, LR, SR, path):

    for i in range(LR.shape[0]):
        vmin0, vmax0 = np.min(SR[i,:,:,0]), np.max(SR[i,:,:,0])
        vmin1, vmax1 = np.min(SR[i,:,:,1]), np.max(SR[i,:,:,1])

        plt.figure(figsize=(12, 12))
        
        plt.subplot(221)
        plt.imshow(LR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
        plt.title('LR 0 Input', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(223)
        plt.imshow(LR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
        plt.title('LR 1 Input', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(222)
        plt.imshow(SR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
        plt.title('SR 0 Output', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(224)
        plt.imshow(SR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
        plt.title('SR 1 Output', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])

        plt.savefig(path+'/img{0:08d}.png'.format(idx[i]), dpi=200, bbox_inches='tight')
        plt.close()