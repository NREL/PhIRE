import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

class Welford:

    def __init__(self):
        self._mean = None
        self._M = None
        self._n = 0


    def update(self, data, axis=None):
        if axis is None:
            axis = tuple(range(data.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)

        data = np.asanyarray(data)
        
        n_b = int(np.prod([data.shape[ax] for ax in axis]))
        n_ab = self._n + n_b

        if self._mean is None:
           self._mean = np.mean(data, axis=axis)
           self._M = np.sum((data - np.expand_dims(self._mean, axis))**2, axis)
        
        else:
            mean_b = np.mean(data, axis=axis) 
            M_b = np.sum((data - np.expand_dims(mean_b, axis))**2, axis)

            delta = mean_b - self._mean
            if abs(self._n - n_b) <= 100 and self._n >= 1000:
                self._mean = (self._n * self._mean + n_b * mean_b) / n_ab
            else:
                self._mean += delta * n_b/n_ab

            self._M += M_b + delta**2 * self._n * n_b / n_ab

        self._n += n_b


    @property
    def var(self):
        assert self._mean is not None
        return self._M / self._n


    @property
    def std(self):
        return np.sqrt(self.var)


    @property
    def mean(self):
        assert self._mean is not None
        return self._mean


    @property
    def n(self):
        return self._n


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def json_to_tf1(json_str):
    json_str = json_str.replace('"groups": 1,', '')
    json_str = json_str.replace('"class_name": "Functional"', '"class_name": "Model"')
    json_str = json_str.replace('"class_name": "HeNormal"', '"class_name": "RandomNormal"')
    return json_str

def downscale_image(x, K):
    tf.reset_default_graph()

    if x.ndim == 3:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

    x_in = tf.placeholder(tf.float64, [None, x.shape[1], x.shape[2], x.shape[3]])

    weight = tf.constant(1.0/K**2, shape=[K, K, x.shape[3], x.shape[3]], dtype=tf.float64)
    downscaled = tf.nn.conv2d(x_in, filter=weight, strides=[1, K, K, 1], padding='SAME')

    with tf.Session() as sess:
        ds_out = sess.run(downscaled, feed_dict={x_in: x})

    return ds_out

def generate_TFRecords(filename, data, mode='test', K=None):
    '''
        Generate TFRecords files for model training or testing

        inputs:
            filename - filename for TFRecord (should by type *.tfrecord)
            data     - numpy array of size (N, h, w, c) containing data to be written to TFRecord
            model    - if 'train', then data contains HR data that is coarsened k times 
                       and both HR and LR data written to TFRecord
                       if 'test', then data contains LR data 
            K        - downscaling factor, must be specified in training mode

        outputs:
            No output, but .tfrecord file written to filename
    '''
    if mode == 'train':
        assert K is not None, 'In training mode, downscaling factor K must be specified'
        data_LR = downscale_image(data, K)

    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in range(data.shape[0]):
            if mode == 'train':
                h_HR, w_HR, c = data[j, ...].shape
                h_LR, w_LR, c = data_LR[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                   'data_HR': _bytes_feature(data[j, ...].tostring()),
                                      'h_HR': _int64_feature(h_HR),
                                      'w_HR': _int64_feature(w_HR),
                                         'c': _int64_feature(c)})
            elif mode == 'test':
                h_LR, w_LR, c = data[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                         'c': _int64_feature(c)})

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString()) 
