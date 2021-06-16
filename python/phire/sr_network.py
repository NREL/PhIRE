''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
import tensorflow as tf
from .layers import *


class SR_NETWORK(object):
    def __init__(self, x_LR=None, x_HR=None, r=None, status='pretraining', alpha_advers=0.001, alpha_content=1.0, encoder=None):

        status = status.lower()
        if status not in ['pretraining', 'training', 'testing']:
            print('Error in network status.')
            exit()

        self.x_LR, self.x_HR = x_LR, x_HR
        self.alpha_advers = alpha_advers
        self.alpha_content = alpha_content

        if encoder:
            self.encoder = encoder()
        else:
            self.encoder = None
        

        if r is None:
            print('Error in SR scaling. Variable r must be specified.')
            exit()

        if status in ['pretraining', 'training']:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=True)
        else:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=False)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        if status == 'pretraining':
            self.g_loss = self.compute_losses(self.x_HR, self.x_SR, None, None, isGAN=False)

            self.d_loss, self.disc_HR, self.disc_SR, self.d_variables = None, None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None

        elif status == 'training':
            self.disc_HR = self.discriminator(self.x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            loss_out = self.compute_losses(self.x_HR, self.x_SR, self.disc_HR, self.disc_SR, isGAN=True)
            self.g_loss = loss_out[0]
            self.d_loss = loss_out[1]
            self.advers_perf = loss_out[2]
            self.content_loss = loss_out[3]
            self.g_advers_loss  = loss_out[4]

        else:
            self.g_loss, self.d_loss = None, None
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None

    def generator(self, x, r, is_training=False, reuse=False):
        if is_training:
            N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
        else:
            N, h, w, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.get_shape()[3]

        k, stride = 3, 1
        output_shape = [N, h+2*k, w+2*k, -1]

        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('in_conv'):
                C_in, C_out = C, 64
                output_shape[-1] = C_out
                x = conv_layer_2d(x, [9, 9, C_in, C_out], stride, k)
                x = tf.nn.leaky_relu(x)

            skip_connection = x

            # B residual blocks
            C_in, C_out = C_out, 64
            output_shape[-1] = C_out
            for i in range(16):
                B_skip_connection = x

                with tf.variable_scope('block_{}a'.format(i+1)):
                    x = conv_layer_2d(x, [k, k, C_in, C_out], stride)
                    #x = bn_layer(x, trainable=is_training)
                    x = tf.nn.leaky_relu(x)

                with tf.variable_scope('block_{}b'.format(i+1)):
                    x = conv_layer_2d(x, [k, k, C_in, C_out], stride)
                    #x = bn_layer(x, trainable=is_training)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('mid_conv'):
                x = conv_layer_2d(x, [k, k, C_in, C_out], stride)
                #x = bn_layer(x, trainable=is_training)
                x = tf.add(x, skip_connection)

            
            # sub-pixel convolutions with PixelShuffle
            r_prod = 1
            for i, r_i in enumerate(r):
                C_out = (r_i**2)*C_in
                with tf.variable_scope('subpixel_conv{}'.format(i+1)):
                    output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C_out]
                    x = conv_layer_2d(x, [k,k,C_in,C_out], stride)
                    x = tf.depth_to_space(x, r_i)
                    x = tf.nn.leaky_relu(x)

                r_prod *= r_i
            """
            with tf.variable_scope('upsample1'):
                #x = bilinear_conv_layer(x, 2, trainable=False)
                H,W= tf.shape(x)[1], tf.shape(x)[2]
                x = tf.image.resize_images(x, (2*H,2*W))

            with tf.variable_scope('upsample2'):
                #x = bilinear_conv_layer(x, 2, trainable=False)
                H,W= tf.shape(x)[1], tf.shape(x)[2]
                x = tf.image.resize_images(x, (2*H,2*W))
            """

            with tf.variable_scope('out_conv'):
                x = conv_layer_2d(x, [9, 9, C_in, C], stride)

        return x


    def discriminator(self, x, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer_2d(x, [3, 3, C, 32], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv2'):
                x = conv_layer_2d(x, [3, 3, 32, 32], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv3'):
                x = conv_layer_2d(x, [3, 3, 32, 64], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv4'):
                x = conv_layer_2d(x, [3, 3, 64, 64], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv5'):
                x = conv_layer_2d(x, [3, 3, 64, 128], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv6'):
                x = conv_layer_2d(x, [3, 3, 128, 128], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv7'):
                x = conv_layer_2d(x, [3, 3, 128, 256], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv8'):
                x = conv_layer_2d(x, [3, 3, 256, 256], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = flatten_layer(x)
            with tf.variable_scope('fully_connected1'):
                x = dense_layer(x, 1024)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('fully_connected2'):
                x = dense_layer(x, 1)

        return x

    def compute_losses(self, x_HR, x_SR, d_HR, d_SR, alpha_advers=0.001, isGAN=False):
        
        if self.encoder is not None:
            r1, r2 = self.encoder(x_HR, training=False), self.encoder(x_SR, training=False)
            diff = r1 - r2
        else:
            diff = x_HR - x_SR

        beta = 1.5
        alpha_tv = 0#5e-3
        img_grad_y_l2_sqr = tf.math.squared_difference(x_SR[:, 1:, :-1, :], x_SR[:,:-1,:-1,:])
        img_grad_x_l2_sqr = tf.math.squared_difference(x_SR[:, :-1, 1:, :], x_SR[:,:-1,:-1,:])
        tv_reg = tf.math.reduce_mean((img_grad_y_l2_sqr + img_grad_x_l2_sqr)**(beta/2), axis=[1,2,3])

        content_loss = tf.reduce_mean(diff**2, axis=[1, 2, 3])

        if isGAN:
            g_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR))

            d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                                                                    labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

            advers_perf = [tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) > 0.5, tf.float32)), # % true positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) < 0.5, tf.float32)), # % true negative
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) > 0.5, tf.float32)), # % false positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) < 0.5, tf.float32))] # % false negative

            g_loss = self.alpha_content*tf.reduce_mean(content_loss) + self.alpha_advers*tf.reduce_mean(g_advers_loss) #+ alpha_tv*tf.reduce_mean(tv_reg)
            d_loss = tf.reduce_mean(d_advers_loss)

            return g_loss, d_loss, advers_perf, content_loss, g_advers_loss
        else:
            return self.alpha_content*tf.reduce_mean(content_loss) #+ alpha_tv*tf.reduce_mean(tv_reg)
    
