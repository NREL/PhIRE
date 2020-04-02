''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
import tensorflow as tf
import numpy as np
import sys
sys.path.append('utils')
from layer import *

class SRGAN(object):
    def __init__(self, x_LR=None, x_HR=None, r=None, status='pre-training', alpha_adverse = 0.001):

        status = status.lower()
        if status not in ['pre-training', 'training', 'testing']:
            print('Error in network status.')
            exit()

        self.x_LR = x_LR
        if x_HR is not None:
            self.x_HR = x_HR

        if r is None:
            print('Error in SR scaling. Variable r must be specified.')
            exit()

        if status in ['pre-training', 'training']:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=True)
        else:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=False)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        if status == 'pre-training':
            self.g_loss = self.compute_losses(x_HR, self.x_SR, None, None, alpha_adverse, isGAN=False)

        elif status == 'training':
            self.disc_HR = self.discriminator(x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            loss_out = self.compute_losses(x_HR, self.x_SR, self.disc_HR, self.disc_SR, alpha_adverse, isGAN=True)
            self.g_loss = loss_out[0]
            self.d_loss = loss_out[1]
            self.advers_perf = loss_out[2]
            self.p_loss = loss_out[3]
            self.g_ad_loss  = loss_out[4]
        else:
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None

            self.g_loss, self.d_loss = None, None


    def generator(self, x, r=None, is_training=False, reuse=False):
        if is_training:
            N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
        else:
            N, h, w, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.get_shape()[3]

        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                C_in, C_out = C, 64
                x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)
                x = tf.nn.relu(x)

            skip_connection = x

            # B residual blocks
            C_in, C_out = C_out, 64
            for i in range(16):
                B_skip_connection = x

                with tf.variable_scope('block_{}a'.format(i+1)):
                    x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)
                    x = tf.nn.relu(x)

                with tf.variable_scope('block_{}b'.format(i+1)):
                    x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('deconv2'):
                x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, h, w, C_out], 1)
                x = tf.add(x, skip_connection)

            # Super resolution scaling
            r_prod = 1
            for i, r_i in enumerate(r):
                C_out = (r_i**2)*C_in
                with tf.variable_scope('deconv{}'.format(i+3)):
                    x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, r_prod*h, r_prod*w, C_out], 1)
                    x = tf.depth_to_space(x, r_i)
                    x = tf.nn.relu(x)

                r_prod *= r_i

            C_out = C
            with tf.variable_scope('deconv_out'):
                x = deconv_layer_2d(x, [3, 3, C_out, C_in], [N, r_prod*h, r_prod*w, C_out], 1)

        return x


    def discriminator(self, x, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]

        # alternate the stride between 1 and 2 every other layer
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer_2d(x, [3, 3, C, 32], 1)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = conv_layer_2d(x, [3, 3, 32, 32], 2)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = conv_layer_2d(x, [3, 3, 32, 64], 1)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                x = conv_layer_2d(x, [3, 3, 64, 64], 2)
                x = lrelu(x)

            with tf.variable_scope('conv5'):
                x = conv_layer_2d(x, [3, 3, 64, 128], 1)
                x = lrelu(x)

            with tf.variable_scope('conv6'):
                x = conv_layer_2d(x, [3, 3, 128, 128], 2)
                x = lrelu(x)

            with tf.variable_scope('conv7'):
                x = conv_layer_2d(x, [3, 3, 128, 256], 1)
                x = lrelu(x)

            with tf.variable_scope('conv8'):
                x = conv_layer_2d(x, [3, 3, 256, 256], 2)
                x = lrelu(x)

            x = flatten_layer(x)
            with tf.variable_scope('fully_connected1'):
                x = dense_layer(x, 1024)
                x = lrelu(x)

            with tf.variable_scope('fully_connected2'):
                x = dense_layer(x, 1)

        return x


    def downscale_image(self, x, K):
        """Downsale the iamge by a factor of K."""
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled


    def compute_losses(self, x_HR, x_SR, d_HR, d_SR, alpha_adverse = 0.001, isGAN=False):
        """Compute the losses for the generator and discriminator networks"""

        def compute_perceptual_loss(x_HR, x_SR):
            N_batch = tf.shape(x_HR)[0]
            content_loss = tf.reduce_mean((x_HR - x_SR)**2, axis=[1, 2, 3])
            
            return content_loss


        def compute_adversarial_loss(d_HR, d_SR):
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR))

            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                                                             labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))
            d_loss = tf.reduce_mean(d_loss)

            return g_loss, d_loss

        def compute_adversarial_performance(d_HR, d_SR):
            adv_TP = tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) > 0.5, tf.float32))
            adv_TN = tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) < 0.5, tf.float32))
            adv_FP = tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) > 0.5, tf.float32))
            adv_FN = tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) < 0.5, tf.float32))

            return adv_TP, adv_TN, adv_FP, adv_FN

        percept_loss = compute_perceptual_loss(x_HR, x_SR)

        if isGAN:
            g_advers_loss, d_advers_loss = compute_adversarial_loss(d_HR, d_SR)
            advers_perf = compute_adversarial_performance(d_HR, d_SR)

            g_loss = tf.reduce_mean(percept_loss) + alpha_adverse*tf.reduce_mean(g_advers_loss)
            d_loss = d_advers_loss

            return g_loss, d_loss, advers_perf, percept_loss, g_advers_loss
        else:

            return tf.reduce_mean(percept_loss)
