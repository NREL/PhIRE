import tensorflow as tf
import numpy as np


def conv_layer_2d(x, filter_shape, stride, trainable=True):
    W = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),#tf.contrib.layers.variance_scaling_initializer(),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[filter_shape[-1]],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    x = tf.nn.bias_add(tf.nn.conv2d(
        input=x,
        filter=W,
        strides=[1, stride, stride, 1],
        padding='SAME'), b)

    return x

def deconv_layer_2d(x, filter_shape, output_shape, stride, trainable=True):
    x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
    W = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),#tf.contrib.layers.variance_scaling_initializer(),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[output_shape[-1]],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    x = tf.nn.bias_add(tf.nn.conv2d_transpose(
        value=x,
        filter=W,
        output_shape=output_shape,
        strides=[1, stride, stride, 1],
        padding='SAME'), b)

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


def upsample_filter(size, dtype='f4'):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    From https://github.com/shelhamer/fcn.berkeleyvision.org
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    W = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)

    print(W.shape)
    return W.astype(dtype)


def bilinear_conv_layer(x, factor, trainable=True):
    B,H,W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    C = x.shape[-1]
    
    C_int = int(C)

    bilinear_filter = np.zeros((2*factor, 2*factor, C_int, C_int), dtype='f4')
    bilinear_filter[:,:,range(C_int), range(C_int)] = upsample_filter(2*factor)[:,:,None]
    kernel = tf.get_variable(
        name='weight',
        #shape=(2*factor, 2*factor, C, C),
        dtype=tf.float32,
        initializer=bilinear_filter,
        trainable=trainable
    )
    output_shape= [B,factor*H,factor*W,C]
    y = tf.nn.conv2d_transpose(
        x,
        filter=kernel,
        output_shape=output_shape,
        strides=[1, factor, factor, 1],
        padding='SAME'
    )

    print(y.shape)

    return y


def bn_layer(x, trainable=True):
    C = x.shape[-1]

    offset = tf.get_variable(
        name='offset',
        shape=[C],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable
    )

    scale = tf.get_variable(
        name='scale',
        shape=[C],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=trainable
    )

    mean, variance = tf.nn.moments(x, axes=[0,1,2])
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 0.001)