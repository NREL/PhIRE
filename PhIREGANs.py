''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from itertools import count
from time import strftime, time
from utils import plot_SR_data
from sr_network import SR_NETWORK


class PhIREGANs:
    # Network training meta-parameters
    DEFAULT_N_EPOCHS = 10 # Number of epochs of training
    DEFAULT_LEARNING_RATE = 1e-4 # Learning rate for gradient descent (may decrease to 1e-5 after initial training)
    DEFAULT_EPOCH_SHIFT = 0 # If reloading previously trained network, what epoch to start at
    DEFAULT_SAVE_EVERY = 10 # How frequently (in epochs) to save model weights
    DEFAULT_PRINT_EVERY = 2 # How frequently (in iterations) to write out performance
    N_SHUFFLE = 4000

    def __init__(self, data_type, N_epochs=None, learning_rate=None, epoch_shift=None, save_every=None, print_every=None, mu_sig=None, alpha_content=1.0, encoder=None):

        self.N_epochs      = N_epochs if N_epochs is not None else self.DEFAULT_N_EPOCHS
        self.learning_rate = learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE
        self.epoch_shift   = epoch_shift if epoch_shift is not None else self.DEFAULT_EPOCH_SHIFT
        self.save_every    = save_every if save_every is not None else self.DEFAULT_SAVE_EVERY
        self.print_every   = print_every if print_every is not None else self.DEFAULT_PRINT_EVERY

        self.data_type = data_type
        self.mu_sig = mu_sig
        self.LR_data_shape = None

        self.alpha_content = alpha_content
        self.encoder = encoder

        # Set various paths for where to save data
        self.run_id        = '-'.join([self.data_type, strftime('%Y%m%d-%H%M%S')])
        self.model_name    = '/'.join(['models', self.run_id])
        self.data_out_path = '/'.join(['data_out', self.run_id])

    def setSave_every(self, in_save_every):
        self.save_every = in_save_every

    def setPrint_every(self, in_print_every):
        self.print_every = in_print_every

    def setEpochShift(self, shift):
        self.epoch_shift = shift

    def setNum_epochs(self, in_epochs):
        self.N_epochs = in_epochs

    def setLearnRate(self, learn_rate):
        self.learning_rate = learn_rate

    def setModel_name(self, in_model_name):
        self.model_name = in_model_name

    def set_data_out_path(self, in_data_path):
        self.data_out_path = in_data_path
    
    def reset_run_id(self):
        self.run_id        = '-'.join([self.data_type, strftime('%Y%m%d-%H%M%S')])
        self.model_name    = '/'.join(['models', self.run_id])
        self.data_out_path = '/'.join(['data_out', self.run_id])

    def pretrain(self, r, data_path, model_path=None, batch_size=100):
        '''
            This method trains the generator without using a disctiminator/adversarial training. 
            This method should be called to sufficiently train the generator to produce decent images before 
            moving on to adversarial training with the train() method.

            inputs:
                r          - (int array) should be array of prime factorization of amount of super-resolution to perform
                data_path  - (string) path of training data file to load in
                model_path - (string) path of previously trained model to load in if continuing training
                batch_size - (int) number of images to grab per batch. decrease if running out of memory

            output:
                saved_model - (string) path to the trained model
        '''
        
        tf.reset_default_graph()
        
        if self.mu_sig is None:
            self.set_mu_sig(data_path, batch_size)
        
        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        x_LR = tf.placeholder(tf.float32, [None, h,             w,            C])
        x_HR = tf.placeholder(tf.float32, [None, h*np.prod(r),  w*np.prod(r), C])

        model = SR_NETWORK(x_LR, x_HR, r=r, status='pretraining', alpha_content=self.alpha_content, encoder=self.encoder)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_train_op = optimizer.minimize(model.g_loss, var_list= model.g_variables)
        init = tf.variables_initializer(model.g_variables + optimizer.variables())
        

        # Create saver and checkpoint dir
        checkpoint_dir = '/'.join([self.model_name, 'pretraining'])
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass
        checkpoint_name = checkpoint_dir + '/generator'
        last_checkpoint = None

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        print('Done.')

        print('Building data pipeline ...', end=' ')
        idx, LR_out, HR_out, init_iter = self.make_train_ds(data_path, batch_size)
        print('Done.')

        with tf.keras.backend.get_session() as sess:
            print('Training network ...')
            sys.stdout.flush()

            sess.run(init)

            if model_path is not None:
                print('Loading previously trained network...', end=' ')
                g_saver.restore(sess, model_path)
                print('Done.')

            # Start training
            iters = 0
            for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
                print('Epoch: %d' %(epoch))
                start_time = time()

                sess.run(init_iter)
                try:
                    epoch_loss, N = 0, 0
                    while True:
                        batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])
                        N_batch = batch_LR.shape[0]
                        feed_dict = {x_HR:batch_HR, x_LR:batch_LR}

                        # Training step of the generator
                        sess.run(g_train_op, feed_dict=feed_dict)

                        # Calculate current losses
                        gl = sess.run(model.g_loss, feed_dict={x_HR: batch_HR, x_LR: batch_LR})

                        epoch_loss += gl*N_batch
                        N += N_batch

                        iters += 1
                        if (iters % self.print_every) == 0:
                            print('Iteration=%d, G loss=%.5f' %(iters, gl))
                            sys.stdout.flush()

                except tf.errors.OutOfRangeError:
                    pass

                if epoch % self.save_every == 0:
                    last_checkpoint = g_saver.save(sess, checkpoint_name, global_step=epoch)

                epoch_loss = epoch_loss/N

                print('Epoch generator training loss=%.5f' %(epoch_loss))
                print('Epoch took %.2f seconds\n' %(time() - start_time))
                sys.stdout.flush()

            if epoch % self.save_every != 0:
                last_checkpoint = g_saver.save(sess, checkpoint_name, global_step=epoch)

            print('Done.')
            return last_checkpoint

    def train(self, r, data_path, model_path, batch_size=100, alpha_advers=0.001):
        '''
            This method trains the generator using a disctiminator/adversarial training. 
            This method should be called after a sufficiently pretrained generator has been saved.

            inputs:
                r            - (int array) should be array of prime factorization of amount of super-resolution to perform
                data_path    - (string) path of training data file to load in
                model_path   - (string) path of previously pretrained or trained model to load
                batch_size   - (int) number of images to grab per batch. decrease if running out of memory
                alpha_advers - (float) scaling value for the effect of the discriminator

            output:
                g_saved_model - (string) path to the trained generator model
        '''
        
        tf.reset_default_graph()

        assert model_path is not None, 'Must provide path for pretrained model'
        
        if self.mu_sig is None:
            self.set_mu_sig(data_path, batch_size)
        
        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        x_LR = tf.placeholder(tf.float32, [None, h,             w,            C])
        x_HR = tf.placeholder(tf.float32, [None, h*np.prod(r),  w*np.prod(r), C])

        model = SR_NETWORK(x_LR, x_HR, r=r, status='training', alpha_advers=alpha_advers, alpha_content=self.alpha_content, encoder=self.encoder)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_train_op = optimizer.minimize(model.g_loss, var_list=model.g_variables)
        d_train_op = optimizer.minimize(model.d_loss, var_list=model.d_variables)
        init = tf.variables_initializer(model.g_variables + model.d_variables + optimizer.variables())

        # Create savera and checkpoint dir
        checkpoint_dir = '/'.join([self.model_name, 'training'])
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass
        checkpoint_name = checkpoint_dir + '/gan'
        last_checkpoint = None

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        gd_saver = tf.train.Saver(var_list=(model.g_variables+model.d_variables), max_to_keep=10000)
        print('Done.')

        print('Building data pipeline ...', end=' ')
        idx, LR_out, HR_out, init_iter = self.make_train_ds(data_path, batch_size)
        print('Done.')

        with tf.keras.backend.get_session() as sess:
            print('Training network ...')
            sys.stdout.flush()

            sess.run(init)

            print('Loading previously trained network...', end=' ')
            try:
                gd_saver.restore(sess, model_path)
                print('Done. Restored both generator & discriminator')
            except tf.errors.NotFoundError:
                g_saver.restore(sess, model_path)
                print('Done. Restored generator only')
            sys.stdout.flush()

            # Start training
            iters = 0
            for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
                print('Epoch: '+str(epoch))
                start_time = time()

                # Loop through training data
                sess.run(init_iter)
                try:
                    epoch_g_loss, epoch_d_loss, N = 0, 0, 0
                    while True:
                        batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])
                        N_batch = batch_LR.shape[0]
                        feed_dict = {x_HR:batch_HR, x_LR:batch_LR}

                        # Initial training of the discriminator and generator
                        sess.run(d_train_op, feed_dict=feed_dict)
                        sess.run(g_train_op, feed_dict=feed_dict)

                        # Calculate current losses
                        gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)

                        gen_count = 1
                        while (dl < 0.460) and gen_count < 2:#30:
                            # Discriminator did too well -> train the generator extra
                            sess.run(g_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)
                            gen_count += 1

                        dis_count = 1
                        while (dl > 0.6) and dis_count < 2:#30:
                            # Generator fooled the discriminator -> train the discriminator extra
                            sess.run(d_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)
                            dis_count += 1

                        epoch_g_loss += gl*N_batch
                        epoch_d_loss += dl*N_batch
                        N += N_batch

                        iters += 1
                        if (iters % self.print_every) == 0:
                            g_cl, g_al = sess.run([model.content_loss, model.g_advers_loss], feed_dict=feed_dict)

                            print('Number of generator training steps=%d, Number of discriminator training steps=%d, ' %(gen_count, dis_count))
                            print('G loss=%.5f, Content component=%.5f, Adversarial component=%.5f' %(gl, np.mean(g_cl), np.mean(g_al)))
                            print('D loss=%.5f' %(dl))
                            print('TP=%.5f, TN=%.5f, FP=%.5f, FN=%.5f' %(p[0], p[1], p[2], p[3]))
                            print('')
                            sys.stdout.flush()

                except tf.errors.OutOfRangeError:
                    pass

                if epoch % self.save_every == 0:                    
                    last_checkpoint = gd_saver.save(sess, checkpoint_name, global_step=epoch)

                g_loss = epoch_g_loss/N
                d_loss = epoch_d_loss/N

                print('Epoch generator training loss=%.5f, discriminator training loss=%.5f' %(g_loss, d_loss))
                print('Epoch took %.2f seconds\n' %(time() - start_time))
                sys.stdout.flush()

            if epoch % self.save_every != 0:                    
                last_checkpoint = gd_saver.save(sess, checkpoint_name, global_step=epoch)

        print('Done.')
        return last_checkpoint


    def test(self, r, data_path, model_path, batch_size=100, plot_data=False, save_every=1, return_batches=False, return_hr=False):
        '''
            This method loads a previously trained model and runs it on test data

            inputs:
                r          - (int array) should be array of prime factorization of amount of super-resolution to perform
                data_path  - (string) path of test data file to load in
                model_path - (string) path of model to load in
                batch_size - (int) number of images to grab per batch. decrease if running out of memory
                plot_data  - (bool) flag for whether or not to plot LR and SR images
        '''

        tf.reset_default_graph()
        
        assert self.mu_sig is not None, 'Value for mu_sig must be set first.'
        
        self.set_LR_data_shape(data_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        x_LR = tf.placeholder(tf.float32, [None, None, None, C])
        model = SR_NETWORK(x_LR, r=r, status='testing', alpha_content=self.alpha_content, encoder=self.encoder)
        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        print('Done.')
        
        print('Building data pipeline ...', end=' ')
        if return_hr:
            idx, LR_out, HR_out, init_iter = self.make_train_ds(data_path, batch_size, shuffle=False)
        else:
            idx, LR_out, init_iter = self.make_test_ds(data_path, batch_size)
        print('Done.')

        if not os.path.exists(self.data_out_path):
                os.makedirs(self.data_out_path)

        with tf.keras.backend.get_session() as sess:
            print('Loading saved network ...', end=' ')
            g_saver.restore(sess, model_path)
            print('Done.')
            
            with open(self.data_out_path + '/dataSR.npy', 'wb') as out_f:            
                print('Running test data ...')
                sess.run(init_iter)
                try:
                    for i in count():
                        if return_hr:
                            batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])
                        else:
                            batch_idx, batch_LR = sess.run([idx, LR_out])
                        N_batch = batch_LR.shape[0]

                        if i % self.print_every == 0:
                            print('batch ', i)
                            sys.stdout.flush()

                        if i % save_every != 0:
                            continue

                        feed_dict = {x_LR:batch_LR}

                        batch_SR = sess.run(model.x_SR, feed_dict=feed_dict)

                        batch_LR = self.mu_sig[1]*batch_LR + self.mu_sig[0]
                        batch_SR = self.mu_sig[1]*batch_SR + self.mu_sig[0]
                        
                        if plot_data:
                            img_path = '/'.join([self.data_out_path, 'imgs'])
                            if not os.path.exists(img_path):
                                os.makedirs(img_path)
                            plot_SR_data(batch_idx, batch_LR, batch_SR, img_path)

                        if return_batches and return_hr:
                            batch_HR = self.mu_sig[1]*batch_HR + self.mu_sig[0]
                            yield batch_LR, batch_SR, batch_HR
                        elif return_batches:
                            yield batch_LR, batch_SR
                        else:
                            np.save(out_f, batch_SR, allow_pickle=False)

                except tf.errors.OutOfRangeError:
                    pass

        print('Done.')

    def _parse_train_(self, serialized_example, mu_sig=None):
        '''
            Parser data from TFRecords for the models to read in for (pre)training

            inputs:
                serialized_example - batch of data drawn from tfrecord
                mu_sig             - mean, standard deviation if known

            outputs:
                idx     - array of indicies for each sample
                data_LR - array of LR images in the batch
                data_HR - array of HR images in the batch
        '''

        feature = {'index': tf.FixedLenFeature([], tf.int64),
                 'data_LR': tf.FixedLenFeature([], tf.string),
                    'h_LR': tf.FixedLenFeature([], tf.int64),
                    'w_LR': tf.FixedLenFeature([], tf.int64),
                 'data_HR': tf.FixedLenFeature([], tf.string),
                    'h_HR': tf.FixedLenFeature([], tf.int64),
                    'w_HR': tf.FixedLenFeature([], tf.int64),
                       'c': tf.FixedLenFeature([], tf.int64)}
        example = tf.parse_single_example(serialized_example, feature)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']
        h_HR, w_HR = example['h_HR'], example['w_HR']

        c = example['c']
  
        data_LR = tf.decode_raw(example['data_LR'], tf.float32)
        data_HR = tf.decode_raw(example['data_HR'], tf.float32)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
        data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

        if False:
            u, std = [275.37314, 1.8918975e-08, 2.3000993e-07], [16.993176, 2.1903368e-05, 4.4884804e-05]
            data_LR = (data_LR - u) / std
            data_HR = (data_HR - u) / std

            data_LR = tf.math.sign(data_LR) * tf.math.log1p(tf.math.abs(data_LR))
            data_HR = tf.math.sign(data_HR) * tf.math.log1p(tf.math.abs(data_HR))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]
            data_HR = (data_HR - mu_sig[0])/mu_sig[1]

        return idx, data_LR, data_HR

    def _parse_test_(self, serialized_example, mu_sig=None):
        '''
            Parser data from TFRecords for the models to read in for testing

            inputs:
                serialized_example - batch of data drawn from tfrecord
                mu_sig             - mean, standard deviation if known

            outputs:
                idx     - array of indicies for each sample
                data_LR - array of LR images in the batch
        '''

        feature = {'index': tf.FixedLenFeature([], tf.int64),
                 'data_LR': tf.FixedLenFeature([], tf.string),
                    'h_LR': tf.FixedLenFeature([], tf.int64),
                    'w_LR': tf.FixedLenFeature([], tf.int64),
                       'c': tf.FixedLenFeature([], tf.int64)}
        example = tf.parse_single_example(serialized_example, feature)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']

        c = example['c']

        data_LR = tf.decode_raw(example['data_LR'], tf.float32)
        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

        if False:
            u, std = [275.37314, 1.8918975e-08, 2.3000993e-07], [16.993176, 2.1903368e-05, 4.4884804e-05]
            data_LR = (data_LR - u) / std
            data_LR = tf.math.sign(data_LR) * tf.math.log1p(tf.math.abs(data_LR))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]

        return idx, data_LR


    def make_train_ds(self, data_path, batch_size, shuffle=True):
        ds = tf.data.TFRecordDataset(data_path, num_parallel_reads=4)
        ds = ds.map(lambda xx: self._parse_train_(xx, self.mu_sig))

        if shuffle:
            ds = ds.shuffle(self.N_SHUFFLE)

        ds = ds.batch(batch_size).prefetch(5)

        iterator = tf.data.Iterator.from_structure(ds.output_types,ds.output_shapes)
        idx, LR_out, HR_out = iterator.get_next()
        init_iter = iterator.make_initializer(ds)

        return idx, LR_out, HR_out, init_iter


    def make_test_ds(self, data_path, batch_size):
        ds = tf.data.TFRecordDataset(data_path, num_parallel_reads=4)
        ds = ds.map(lambda xx: self._parse_test_(xx, self.mu_sig)).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(ds.output_types,
                                                   ds.output_shapes)
        idx, LR_out = iterator.get_next()
        init_iter = iterator.make_initializer(ds)

        return idx, LR_out, init_iter


    def set_mu_sig(self, data_path, batch_size=1):
        '''
            Compute mean (mu) and standard deviation (sigma) for each data channel
            inputs:
                data_path - (string) path to the tfrecord for the training data
                batch_size - number of samples to grab each interation

            outputs:
                sets self.mu_sig
        '''
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train_).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        _, _, HR_out = iterator.get_next()

        with tf.Session() as sess:
            N, mu, sigma = 0, 0, 0
            try:
                while True:
                    data_HR = sess.run(HR_out)

                    N_batch, h, w, c = data_HR.shape
                    N_new = N + N_batch

                    mu_batch = np.mean(data_HR, axis=(0, 1, 2))
                    sigma_batch = np.var(data_HR, axis=(0, 1, 2))

                    sigma = (N/N_new)*sigma + (N_batch/N_new)*sigma_batch + (N*N_batch/N_new**2)*(mu - mu_batch)**2
                    mu = (N/N_new)*mu + (N_batch/N_new)*mu_batch

                    N = N_new

            except tf.errors.OutOfRangeError:
                pass

        self.mu_sig = [mu, np.sqrt(sigma)]

        print('Done.')

    def set_LR_data_shape(self, data_path):
        '''
            Get size and shape of LR input data
            inputs:
                data_path - (string) path to the tfrecord of the data

            outputs:
                sets self.LR_data_shape
        '''
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_test_).batch(1)

        iterator = dataset.make_one_shot_iterator()
        _, LR_out = iterator.get_next()

        with tf.Session() as sess:
            data_LR = sess.run(LR_out)
        
        self.LR_data_shape = data_LR.shape[1:]
