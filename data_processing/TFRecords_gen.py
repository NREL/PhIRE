#options for 2d and 1d
#options for surface roughness
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generateTestTFR():
    tt = 'test'
     with tf.python_io.TFRecordWriter(file_prefix + tt + res_set_LR +  res_set_HR + '.tfrecord') as writer:
        data_LR = np.load('')
        data_HR = np.load(file_prefix + tt + res_set_HR + '.npy')
        N,  h_LR, w_LR, c  = data_LR.shape
        N2, h_HR, w_HR, c2 = data_HR.shape

        assert N == N2, 'Mismatch in number of samples'
        assert c == c2, 'Mismatch in number of channels'
        for j in range(N):
            example = tf.train.Example(features=tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                   'data_HR': _bytes_feature(data_HR[j, ...].tostring()),
                                      'h_HR': _int64_feature(h_HR),
                                      'w_HR': _int64_feature(w_HR),
                                         'c': _int64_feature(c)}))
            writer.write(example.SerializeToString())

    return test

def generateTrainTFR():
    tt = 'train'
     with tf.python_io.TFRecordWriter(file_prefix + tt + res_set_LR + res_set_HR + '.tfrecord') as writer:
        data_LR = np.load('')
        data_HR = np.load(file_prefix + tt + res_set_HR + '.npy')
        N,  h_LR, w_LR, c  = data_LR.shape
        N2, h_HR, w_HR, c2 = data_HR.shape

        assert N == N2, 'Mismatch in number of samples'
        assert c == c2, 'Mismatch in number of channels'
        for j in range(N):
            example = tf.train.Example(features=tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                   'data_HR': _bytes_feature(data_HR[j, ...].tostring()),
                                      'h_HR': _int64_feature(h_HR),
                                      'w_HR': _int64_feature(w_HR),
                                         'c': _int64_feature(c)}))
            writer.write(example.SerializeToString())

    return train

def generateValidTFR(file_prefix, LR_name, LR_set):
    '''
        generates a TFRecord that only has LR data (ie for use in the SRGANs test function) and no corresponding HR images

        inputs:
            file_prefix - where to save the TFRecord to and/or where to load in the LR and MR files from
            LR_set - file name of the LR input
        outputs:
            validation - string of the file name and path to the generated TFRecord
    '''
    tt = 'valid'

    validation = file_prefix + '_' + LR_name + tt + '_.tfrecord'
    with tf.python_io.TFRecordWriter(validation) as writer:
        data_LR = np.load(LR_set)
        N,  h_LR, w_LR, c  = data_LR.shape

        for j in range(N):
            example = tf.train.Example(features=tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                         'c': _int64_feature(c)}))
            writer.write(example.SerializeToString())

    return validation

if __name__ == '__main__':
    file_prefix = 'wtk_4hrRand_us_2007-2013_ua-va_slices_'
    LR_set = 'wtk_4hrRand_us_2007-2013_ua-va_slices_test_LR_10.npy'
    HR_set = 'wtk_4hrRand_us_2007-2013_ua-va_slices_test_MR_100.npy'
    val_set = ''
    validation_set = generateValidTFR(file_prefix, 'validation_LR', val_set)
