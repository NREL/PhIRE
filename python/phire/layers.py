import tensorflow as tf
import numpy as np


def gaussian_blur(x, kernel_size, sigma):
    if kernel_size % 2 == 1:
        kernel_x = tf.range(-(kernel_size//2), kernel_size//2 + 1, dtype=tf.float32)
    else:
        a = tf.range(-kernel_size//2, 0, dtype=tf.float32) 
        b = tf.range(1, kernel_size//2 + 1, dtype=tf.float32)
        kernel_x = tf.concat([a,b], axis=0)

    kernel_1d = tf.nn.softmax(-kernel_x**2 / (2.0 * sigma**2))
    kernel_2d = tf.matmul(kernel_1d[:, None], kernel_1d[None, :])

    C = x.get_shape()[-1]
    kernel = tf.tile(kernel_2d[:,:,None,None], (1,1,C,1))
    kernel = tf.stop_gradient(kernel)

    x = tf.pad(x, [[0,0], [kernel_size//2, kernel_size//2], [kernel_size//2, kernel_size//2], [0,0]], 'SYMMETRIC')
    return tf.nn.depthwise_conv2d(x, kernel, (1,1,1,1), 'VALID')


@tf.custom_gradient
def blur_gradient(x, sigma):
    
    def grad(dy):
        g = gaussian_blur(dy, 4, sigma)
        return g, 0.0  # no gradient for sigma

    return x, grad


def checkboard_free_xavier_initializer(r):
    xavier = tf.contrib.layers.xavier_initializer()
    def init(shape, dtype, partition_info):
        O = shape[-1] // (r**2)
        weights = xavier(shape, dtype)
        W_0 = weights[:,:,:,0:O]

        return tf.tile(W_0, [1,1,1,r**2])

    return init


def nn_resize_conv(x, filter_shape, r, stride, trainable=True):
    h,w = x.get_shape()[1], x.get_shape()[2]
    if h.value is None:
        h = tf.shape(x)[1]
    if w.value is None:
        w = tf.shape(x)[2]

    x = tf.image.resize(x, (h*r, w*r), 'nearest')

    W = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)

    b = tf.get_variable(
        name='bias',
        shape=[filter_shape[-1]],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    
    x = tf.nn.conv2d(
        input=x,
        filter=W,
        strides=[1, stride, stride, 1],
        padding='SAME')
    x = tf.nn.bias_add(x, b)

    return x


def subpixel_conv(x, filter_shape, r, stride, trainable=True):
    K1,K2,I,O = filter_shape
    
    W = tf.get_variable(
        name='weight',
        shape=(K1,K2,I, O * r**2),
        dtype=tf.float32,
        initializer=checkboard_free_xavier_initializer(r),
        trainable=trainable)

    b = tf.get_variable(
        name='bias',
        shape=[O * r**2],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    
    x = tf.nn.conv2d(
        input=x,
        filter=W,
        strides=[1, stride, stride, 1],
        padding='SAME')
    
    x = tf.nn.bias_add(x, b)
    x = tf.depth_to_space(x, r)

    return x


def conv_layer_2d(x, filter_shape, stride, trainable=True, kernel_initializer=None):
    kernel_init = kernel_initializer or tf.contrib.layers.xavier_initializer()

    W = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=kernel_init,
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[filter_shape[-1]],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)

    x = tf.nn.conv2d(
        input=x,
        filter=W,
        strides=[1, stride, stride, 1],
        padding='SAME')
    x = tf.nn.bias_add(x, b)

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

    x = tf.nn.conv2d_transpose(
        value=x,
        filter=W,
        output_shape=output_shape,
        strides=[1, stride, stride, 1],
        padding='SAME')
    x = tf.nn.bias_add(x, b)

    return x[:, 3:-3, 3:-3, :]


def flatten_layer(x):
    H,W,C = x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
    x = tf.reshape(x, [-1, H*W*C])
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


def bn_layer(x, momentum=0.99, trainable=True):
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

    rolling_mean = tf.get_variable(
        name='rolling_mean',
        shape=[C],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable
    )
    rolling_var = tf.get_variable(
        name='rolling_var',
        shape=[C],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=trainable
    )

    mean, variance = tf.nn.moments(x, axes=[0,1,2])

    if trainable:
        update_mean = rolling_mean.assign(momentum*rolling_mean + (1-momentum)*mean, read_value=False)
        update_var = rolling_var.assign(momentum*rolling_var + (1-momentum)*variance, read_value=False)
    
        with tf.control_dependencies([update_mean, update_var]):
            y = tf.nn.batch_normalization(x, mean, variance, offset, scale, 0.001)

    else:
        y = tf.nn.batch_normalization(x, rolling_mean, rolling_var, offset, scale, 0.001)

    return y