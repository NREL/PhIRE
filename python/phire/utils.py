import numpy as np
import tensorflow as tf
import xarray as xr


def sliding_window_view(
    x, window_shape, axis=None, *, subok=False, writeable=False
):
    """
    Create a sliding window view into the array with the given window shape.
    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.
    .. versionadded:: 1.20.0
    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.
    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.
    """
    from numpy.core.numeric import normalize_axis_tuple

    window_shape = (
        tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    )
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError("window shape cannot be larger than input array shape")
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return np.lib.stride_tricks.as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


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


class CoordTransform:

    def __init__(self, H, W, lat_up = 90, lat_down = -90, lon_left = 0, lon_right = 360):
        self.H = H
        self.W = W
   
        self.lat_up = lat_up
        self.lat_down = lat_down
        self.lon_left = lon_left
        self.lon_right = lon_right

        self.lat_H = self.lat_up - self.lat_down
        self.lon_W = self.lon_right - self.lon_left

        assert self.lat_H > 0
        assert self.lon_W > 0


    def to_yx(self, lat, lon):
        lat = np.asarray(lat, dtype=np.float32)
        lon = np.asarray(lon, dtype=np.float32)

        y = self.H * (90-lat) / self.lat_H
        x = self.W * lon/self.lon_W
        return y,x


    def to_latlon(self, y, x):
        lat = y/self.H * self.lat_H - 90
        lon = x/self.W * self.lon_W
        return lat,lon


def lanczos_weights(window_size, cutoff):
    n = (window_size+1) // 2
    k = np.arange(-n, n+1)
    f_N = 1 / cutoff

    with np.errstate(divide='ignore', invalid='ignore'):  # k=0 results in div by zero
        w_k = np.sin(2*np.pi * k * f_N) / (np.pi * k)
        sigma_factor = np.sin(np.pi*k / n) * n / (np.pi*k)
        weights = w_k * sigma_factor

    # for k = 0
    weights[n] = 2 * f_N

    return weights[1:-1]


def lanczos_filter_xr(arr, window_size, cutoff, dim, center=False):
    weights = lanczos_weights(window_size, cutoff)
    def filter_(x, axis=None):
        return np.sum(x*weights, axis=axis)

    return arr.rolling(dim={dim: window_size}, center=center).reduce(filter_)


def tv(X):
    """
    Computes discrete total-variation given a batch of images with shape HxWxC

    X: array of shape [B,H,W,C]
    """
    A = np.pad(X, ((0,0), (0,1), (0,0), (0,0)), mode='edge')[:,1:,:,:]
    B = np.pad(X, ((0,0), (0,0), (0,1), (0,0)), mode='edge')[:,:,1:,:]
    return np.sum(np.sqrt((A-X)**2 + (B-X)**2), axis=(1,2))