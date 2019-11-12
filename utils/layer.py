import tensorflow as tf

def lrelu(x, trainable=None):
    alpha = 0.2
    return tf.maximum(alpha * x, x)


def prelu(x, trainable=True):
    alpha = tf.get_variable(
        name='alpha',
        shape=x.get_shape()[-1],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def conv_layer_2d(x, filter_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv2d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, 1],
        padding='SAME')


def conv_layer_2d_multi(x, kernel, filter_in_out, stride, trainable=True):
    x_out = None
    for k in kernel:
        with tf.variable_scope('multi-scale_{}'.format(k)):
            xx_out = conv_layer_2d(x, 
                [k, k, filter_in_out[0], filter_in_out[1]], 
                stride, 
                trainable=trainable)
        if x_out is None:
            x_out = xx_out
        else:
            x_out = tf.maximum(x_out, xx_out)
    return x_out


def conv_layer_3d(x, filter_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv3d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, stride, 1],
        padding='SAME')


def conv_layer_3d_multi(x, kernel, filter_in_out, stride, trainable=True):
    x_out = None
    for k in kernel:
        with tf.variable_scope('multi-scale_{}'.format(k)):
            xx_out = conv_layer_3d(x, 
                [k, k, k, filter_in_out[0], filter_in_out[1]], 
                stride, 
                trainable=trainable)
        if x_out is None:
            x_out = xx_out
        else:
            x_out = tf.maximum(x_out, xx_out)
    return x_out


def deconv_layer_2d(x, filter_shape, output_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv2d_transpose(
        value=x,
        filter=filter_,
        output_shape=output_shape,
        strides=[1, stride, stride, 1])


def deconv_layer_2d_multi(x, kernel, filter_in_out, output_shape, stride, trainable=True):
    x_out = None
    for k in kernel:
        with tf.variable_scope('multi-scale_{}'.format(k)):
            xx_out = deconv_layer_2d(x, 
                [k, k, filter_in_out[0], filter_in_out[1]], 
                output_shape,
                stride,
                trainable=trainable)
        if x_out is None:
            x_out = xx_out
        else:
            x_out = tf.maximum(x_out, xx_out)
    return x_out


def deconv_layer_3d(x, filter_shape, output_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv3d_transpose(
        value=x,
        filter=filter_,
        output_shape=output_shape,
        strides=[1, stride, stride, stride, 1])


def deconv_layer_3d_multi(x, kernel, filter_in_out, output_shape, stride, trainable=True):
    x_out = None
    for k in kernel:
        with tf.variable_scope('multi-scale_{}'.format(k)):
            xx_out = deconv_layer_3d(x, 
                [k, k, k, filter_in_out[0], filter_in_out[1]], 
                output_shape,
                stride,
                trainable=trainable)
        if x_out is None:
            x_out = xx_out
        else:
            x_out = tf.maximum(x_out, xx_out)
    return x_out


def max_pooling_layer(x, size, stride):
    return tf.nn.max_pool(
        value=x,
        ksize=[1, size, size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')


def avg_pooling_layer(x, size, stride):
    return tf.nn.avg_pool(
        value=x,
        ksize=[1, size, size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')


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
    return tf.add(tf.matmul(x, W), b)
    

def batch_norm(x, is_training, decay=0.99, epsilon=0.001, trainable=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.02),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    if is_training:
        return bn_train()
    else:
        return bn_inference()


def instance_norm(x):
    mu = tf.expand_dims(tf.expand_dims(tf.reduce_mean(x, axis=(1, 2)), 1), 2)
    sigma = tf.expand_dims(tf.expand_dims(tf.reduce_mean((x - mu)**2, axis=(1, 2)), 1), 2)

    return (x - mu)/tf.sqrt(sigma + 1e-6)


def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        N, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, (N, h, w, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, h, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, w, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (N, h*r, w*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)

