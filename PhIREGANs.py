''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
import os
import numpy as np
import tensorflow as tf
from time import strftime, time
from utils import *
from srgan import SRGAN

# Suppress TensorFlow warnings about not using GPUs because they're annoying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Needed this line to run TensorFlow for some reason
# Would prefer to find another workaround if possible
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class PhIREGANs:
    # Class Attributes/defaults
    # Network training meta-parameters
    DEFAULT_LEARNING_RATE = 1e-5 # Learning rate for gradient descent (may decrease to 1e-5 after initial training)
    DEFAULT_N_EPOCHS = 1000 # Number of epochs of training
    DEFAULT_EPOCH_SHIFT = 0 # If reloading previously trained network, what epoch to start at
    DEFAULT_SAVE_EVERY = 1 # How frequently (in epochs) to save model weights
    DEFAULT_PRINT_EVERY = 1000 # How frequently (in iterations) to write out performance
    DEFAULT_MU_SIG = [0, 0.0]

    DEFAULT_DATA_TYPE = 'wind'

    def __init__(self, num_epochs=None, learn_rate=None, e_shift=None, save=None, print=None, mu_sig=None, d_type=None):

        self.N_epochs = num_epochs if num_epochs is not None else self.DEFAULT_N_EPOCHS
        self.learning_rate = learn_rate if learn_rate is not None else self.DEFAULT_LEARNING_RATE
        self.epoch_shift = e_shift if e_shift is not None else self.DEFAULT_EPOCH_SHIFT
        self.save_every = save if save is not None else self.DEFAULT_SAVE_EVERY
        self.print_every = print if print is not None else self.DEFAULT_PRINT_EVERY
        self.mu_sig = mu_sig if mu_sig is not None else self.DEFAULT_MU_SIG
        self.data_type = d_type if d_type is not None else self.DEFAULT_DATA_TYPE
        self.LR_data_shape = None

        # Set various paths for where to save data
        self.now = strftime('%Y%m%d-%H%M%S')
        self.model_name = '/'.join(['model', self.now])
        self.test_data_path ='data_out/' + self.data_type + '/' + self.model_name

    def setDataType(self, dt):
        self.data_type = dt

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

    def set_test_data_path(self, in_data_path):
        self.test_data_path = in_data_path

    def pre_train(self, r, train_path, test_path, model_path, batch_size=100):
        '''
            This method trains the generator without using a disctiminator/adversarial training. This method should be called to sufficiently train the generator to produce decent images before moving on to adversarial training with the train() method.

            inputs:
                r - (int array) should be array of prime factorization of amount of super-resolution to perform
                train_path - (string) path of training data file to load in
                test_path - (string) path of testing data file to load in
                model_path - (string) path of model to load in
                batch_size - (int) number of images to grab per batch. decrase if running out of memory

            output:
            model_dr - (string) path to the trained model
        '''

        """Pretrain network (i.e., no adversarial component)."""
        tf.reset_default_graph()
        
        if self.mu_sig is None:
            self.set_mu_sig(train_path, batch_size)
        self.set_LR_data_shape(train_path)
        h, w, C = self.LR_data_shape

        scale = np.prod(r)

        print('Initializing network ...', end=' ')
        # Set high- and low-res data place holders. Make sure the sizes match the data
        x_LR = tf.placeholder(tf.float32, [None,  self.LR_data_shape[1],  self.LR_data_shape[2], self.LR_data_shape[3]])
        x_HR = tf.placeholder(tf.float32, [None, self.LR_data_shape[1]*scale,  self.LR_data_shape[2]*scale, self.LR_data_shape[3]])

        # Initialize network and set optimizer
        model = SRGAN(x_LR, x_HR, r=r, status='pre-training')

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_train_op = optimizer.minimize(model.g_loss, var_list= model.g_variables)
        init = tf.global_variables_initializer()

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        print('Done.')

        print('Building data pipeline ...', end=' ')
        ds_train = tf.data.TFRecordDataset(train_path)
        ds_test = tf.data.TFRecordDataset(test_path)

        ds_train = ds_train.map(lambda xx: self._parse_train_(xx, self.mu_sig)).shuffle(1000).batch(batch_size)
        ds_test = ds_test.map(lambda xx: self._parse_train_(xx, self.mu_sig)).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(ds_train.output_types,
                                                   ds_train.output_shapes)
        idx, LR_out, HR_out = iterator.get_next()

        init_iter_train = iterator.make_initializer(ds_train)
        init_iter_test  = iterator.make_initializer(ds_test)
        print('Done.')

        with tf.Session() as sess:
            print('Training network ...')

            sess.run(init)

            # Load previously trained network
            if self.epoch_shift > 0:
                print('Loading pre-trained network...', end=' ')
                g_saver.restore(sess, model_path)
                print('Done.')

            # Start training
            iters = 0
            for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
                print('Epoch: '+str(epoch))
                start_time = time()

                # Loop through training data
                sess.run(init_iter_train)
                try:
                    epoch_g_loss, epoch_d_loss, N_train = 0, 0, 0
                    while True:
                        iters += 1

                        batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])

                        N_batch = batch_LR.shape[0]

                        feed_dict = {x_HR:batch_HR, x_LR:batch_LR}

                        batch_SR = sess.run(model.x_SR, feed_dict=feed_dict)

                        sess.run(g_train_op, feed_dict=feed_dict)
                        gl = sess.run(model.g_loss, feed_dict={x_HR: batch_HR, x_LR: batch_LR})

                        epoch_g_loss += gl*N_batch
                        N_train += N_batch

                        if (iters % self.print_every) == 0:
                            print('Iterations=%d, G loss=%.5f' %(iters, gl))

                except tf.errors.OutOfRangeError:
                    if (epoch % self.save_every) == 0:
                        model_epoch = '/'.join([model_name, 'pretrain{0:05d}'.format(epoch)])
                        if not os.path.exists(model_epoch):
                            os.makedirs(model_epoch)
                        g_saver.save(sess, '/'.join([model_epoch, 'SRGAN_pretrain']))

                    g_loss_train = epoch_g_loss/N_train

                # Loop through test data
                sess.run(init_iter_test)
                try:
                    test_out = None
                    epoch_g_loss, N_test = 0, 0
                    while True:
                        batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])
                        N_batch = batch_LR.shape[0]

                        feed_dict = {x_HR:batch_HR, x_LR:batch_LR}

                        batch_SR, gl = sess.run([model.x_SR, model.g_loss], feed_dict=feed_dict)

                        epoch_g_loss += gl*N_batch
                        N_test += N_batch
                        
                        batch_LR = self.mu_sig[1]*batch_LR + self.mu_sig[0]
                        batch_SR = self.mu_sig[1]*batch_SR + self.mu_sig[0]
                        batch_HR = self.mu_sig[1]*batch_HR + self.mu_sig[0]
                        if (epoch % self.save_every) == 0:
                            for i, b_idx in enumerate(batch_idx):
                                if test_out is None:
                                    test_out = np.expand_dims(batch_SR[i], 0)
                                else:
                                    test_out = np.concatenate((test_out, np.expand_dims(batch_SR[i], 0)), axis=0)

                except tf.errors.OutOfRangeError:
                    g_loss_test = epoch_g_loss/N_test

                    test_out_path = self.test_data_path

                    test_save_path = test_out_path + 'test_SR_epoch{0:05d}'.format(epoch) + '.npy'

                    if not os.path.exists(test_out_path):
                        os.makedirs(test_out_path)
                    if (epoch % save_every) == 0:
                        np.save(test_save_path, test_out)

                    print('Epoch took %.2f seconds\n' %(time() - start_time))

            # Save model after training is completed
            model_dr = '/'.join([self.model_name, 'pretrain', 'SRGAN_pretrain'])
            if not os.path.exists(self.model_name):
                os.makedirs(self.model_name)
            g_saver.save(sess, model_dr)

        print('Done.')
        return model_dr

    def train(self, r, train_path, test_path, model_path, batch_size=100, alpha_adverse=0.001):
        '''
            This method trains the generator using a disctiminator/adversarial training. This method should be called to sufficiently train the generator to produce decent images before moving on to adversarial training with the train() method.

            inputs:
                r               -   (int array) should be array of prime factorization of amount of super-resolution to perform
                train_path      -   (string) path of training data file to load in
                test_path       -   (string) path of testing data file to load in
                model_path      -   (string) path of model to load in
                batch_size      -   (int) number of images to grab per batch. decrase if running out of
                                memory
                alpha_adverse   - scaling value for the effect of the discriminator

            output:
            [g_model_dr, gd_model_dr] - ([string, string]) paths to the trained generator model
                                        (g_model_dr) and the trained generator with the trained descriminator (gd_model_dr)
        '''

        """Train network using GANs. Only run this after model has been sufficiently pre-trained."""
        tf.reset_default_graph()

        assert model_path is not None, 'Must provide path for pretrained model'
        
        if self.mu_sig is None:
            self.set_mu_sig(train_path, batch_size)
        self.set_LR_data_shape(train_path)
        h, w, C = self.LR_data_shape
        
        scale = np.prod(r)

        print('Initializing network ...', end=' ')

        # Set high- and low-res data place holders. Make sure the sizes match the data
        x_LR = tf.placeholder(tf.float32, [None,  self.LR_data_shape[1],  self.LR_data_shape[2], self.LR_data_shape[3]])
        x_HR = tf.placeholder(tf.float32, [None, self.LR_data_shape[1]*scale,  self.LR_data_shape[2]*scale, self.LR_data_shape[3]])

        # Initialize network and set optimizer
        model = SRGAN(x_LR, x_HR, r=r, status='training', alpha_adverse = alpha_adverse)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_train_op = optimizer.minimize(model.g_loss, var_list=model.g_variables)
        d_train_op = optimizer.minimize(model.d_loss, var_list=model.d_variables)
        init = tf.global_variables_initializer()

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        gd_saver = tf.train.Saver(var_list=(model.g_variables+model.d_variables), max_to_keep=10000)
        print('Done.')

        print('Building data pipeline ...', end=' ')
        ds_train = tf.data.TFRecordDataset(train_path)
        ds_test = tf.data.TFRecordDataset(test_path)

        ds_train = ds_train.map(lambda xx: self._parse_train_(xx, self.mu_sig)).shuffle(1000).batch(batch_size)
        ds_test = ds_test.map(lambda xx: self._parse_train_(xx, self.mu_sig)).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(ds_train.output_types,
                                                   ds_train.output_shapes)
        idx, LR_out, HR_out = iterator.get_next()


        init_iter_train = iterator.make_initializer(ds_train)
        init_iter_test  = iterator.make_initializer(ds_test)
        print('Done.')

        with tf.Session() as sess:
            print('Training network ...')

            sess.run(init)

            print('Loading pre-trained network...', end=' ')
            if 'SRGAN-all' in model_path:
                gd_saver.restore(sess, model_path) #continue training the descriminator if it exists.
            else:
                g_saver.restore(sess, model_path)

            print('Done.')

            # Start training
            iters = 0
            for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
                print('Epoch: '+str(epoch))
                start_time = time()

                # Loop through training data
                sess.run(init_iter_train)
                try:
                    epoch_g_loss, epoch_d_loss, N_train = 0, 0, 0
                    while True:
                        iters += 1

                        batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])

                        N_batch = batch_LR.shape[0]

                        feed_dict = {x_HR:batch_HR, x_LR:batch_LR}

                        batch_SR = sess.run(model.x_SR, feed_dict=feed_dict)

                        #train the discriminator
                        sess.run(d_train_op, feed_dict=feed_dict)

                        #train the generator
                        sess.run(g_train_op, feed_dict=feed_dict)

                        #calculate current discriminator losses
                        gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)

                        gen_count = 0
                        while (dl < 0.460) and gen_count < 30:
                            #discriminator did too well. train the generator extra
                            sess.run(g_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)
                            gen_count += 1

                        dis_count = 0
                        while (dl >= 0.6) and dis_count <30:
                            #generator fooled the discriminator. train the discriminator extra
                            sess.run(d_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)
                            dis_count += 1

                        pl, gal = sess.run([model.p_loss, model.g_ad_loss], feed_dict=feed_dict)

                        epoch_g_loss += gl*N_batch
                        epoch_d_loss += dl*N_batch
                        N_train += N_batch
                        if (iters % self.print_every) == 0:
                            print('G loss=%.5f, Perceptual component=%.5f, Adversarial component=%.5f' %(gl, np.mean(pl), np.mean(gal)))
                            print('D loss=%.5f' %(dl))
                            print('TP=%.5f, TN=%.5f, FP=%.5f, FN=%.5f' %(p[0], p[1], p[2], p[3]))
                            print('')

                except tf.errors.OutOfRangeError:
                    if (epoch % self.save_every) == 0:
                        model_epoch = '/'.join([self.model_name, 'SRGAN{0:05d}'.format(epoch)])
                        if not os.path.exists(model_epoch):
                            os.makedirs(model_epoch)
                        g_saver.save(sess, '/'.join([model_epoch, 'SRGAN']))
                    g_loss_train = epoch_g_loss/N_train
                    d_loss_train = epoch_d_loss/N_train

                sess.run(init_iter_test)
                #loop through test data
                try:
                    test_out = None
                    epoch_g_loss, epoch_d_loss, N_test = 0, 0, 0
                    while True:
                        batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])
                        N_batch = batch_LR.shape[0]

                        feed_dict = {x_HR:batch_HR, x_LR:batch_LR}

                        batch_SR = sess.run(model.x_SR, feed_dict=feed_dict)


                        epoch_g_loss += gl*N_batch
                        epoch_d_loss += dl*N_batch
                        N_test += N_batch
                        batch_LR = self.mu_sig[1]*batch_LR + self.mu_sig[0]
                        batch_SR = self.mu_sig[1]*batch_SR + self.mu_sig[0]
                        batch_HR = self.mu_sig[1]*batch_HR + self.mu_sig[0]
                        if (epoch % self.save_every) == 0:

                            #for i, idx in enumerate(batch_LR.shape[0]): # NEED TO INDEX ALL THE DATA
                            for i, b_idx in enumerate(batch_idx):
                                if test_out is None:
                                    test_out = np.expand_dims(batch_SR[i], 0)
                                else:
                                    test_out = np.concatenate((test_out, np.expand_dims(batch_SR[i], 0)), axis=0)

                except tf.errors.OutOfRangeError:

                    g_loss_test = epoch_g_loss/N_test
                    d_loss_test = epoch_d_loss/N_test

                    if not os.path.exists(test_out_path):
                        os.makedirs(test_out_path)
                    if (epoch % self.save_every) == 0:
                        np.save(test_out_path +'test_SR_epoch{0:05d}'.format(epoch)+'.npy', test_out)

                    print('Epoch took %.2f seconds\n' %(time() - start_time))
            g_model_dr, gd_model_dr = '/'.join([self.model_name, 'SRGAN', 'SRGAN']), '/'.join([self.model_name, 'SRGAN-all', 'SRGAN'])
            if not os.path.exists(self.model_name):
                os.makedirs(self.model_name)
            g_saver.save(sess, g_model_dr)
            gd_saver.save(sess, gd_model_dr)

        print('Done.')
        return [g_model_dr, gd_model_dr]

    def test(self, r, train_path, test_path, model_path, batch_size=100):
        '''
            This method loads a previously trained model and runs it on test data

            inputs:
                r           -   (int array) should be array of prime factorization of amount of
                                super-resolution to perform
                train_path  -   (string) path of training data file to load in
                test_path   -   (string) path of test data file to load in
                model_path  -   (string) path of model to load in
                batch_size  -   (int) number of images to grab per batch. decrase if running out of
                                memory
            output:
            LR_out, data_out  - (numpy array, numpy array) arrays of the LR input and corresponding
                                SR output
        '''

        """Run test data through generator and save output."""
        tf.reset_default_graph()
        
        if self.mu_sig is None:
            assert train_path is not None
            self.set_mu_sig(train_path, batch_size)
        self.set_LR_data_shape(test_path)
        h, w, C = self.LR_data_shape

        print('Initializing network ...', end=' ')
        
        x_LR = tf.placeholder(tf.float32, [None, None, None, C])

        # Initialize network
        model = SRGAN(x_LR, r=r, status='testing')

        init = tf.global_variables_initializer()
        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        print('Done.')

        print('Building data pipeline ...', end=' ')

        ds_test = tf.data.TFRecordDataset(test_path)
        ds_test = ds_test.map(lambda xx: self._parse_test_(xx, self.mu_sig)).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(ds_test.output_types,
                                                   ds_test.output_shapes)
        idx, LR_out = iterator.get_next()

        init_iter_test  = iterator.make_initializer(ds_test)
        print('Done.')

        # Set expected size of SR data.
        with tf.Session() as sess:
            print('Loading saved network ...', end=' ')
            sess.run(init)
            g_saver.restore(sess, model_path)
            print('Done.')

            print('Running test data ...')
            # Loop through test data
            sess.run(init_iter_test)
            try:
                data_out = None
                while True:

                    batch_idx, batch_LR = sess.run([idx, LR_out])
                    N_batch = batch_LR.shape[0]

                    feed_dict = {x_LR:batch_LR}

                    batch_SR = sess.run(model.x_SR, feed_dict=feed_dict)

                    batch_SR = self.mu_sig[1]*batch_SR + self.mu_sig[0]
                    if data_out is None:
                        data_out = batch_SR
                    else:
                        data_out = np.concatenate((data_out, batch_SR), axis=0)

            except tf.errors.OutOfRangeError:
                pass

            if not os.path.exists(self.test_data_path):
                os.makedirs(self.test_data_path)
            np.save(self.test_data_path+'/test_SR.npy', data_out)

        print('Done.')

    def _parse_train_(self, serialized_example, mu_sig=None):
        '''
            This method parses TFRecords for the models to read in for training or pretraining.

            inputs:
                serialized_example - should only contain LR images (no HR ground truth images)
                mu_sig             - mean, sigma if known

            outputs:
                idx     - an array of indicies for each sample
                data_LR - array of LR images in the batch
                data_HR - array of HR (corresponding ground truth HR version of data_LR)
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

        data_LR = tf.decode_raw(example['data_LR'], tf.float64)
        data_HR = tf.decode_raw(example['data_HR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))
        data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]
            data_HR = (data_HR - mu_sig[0])/mu_sig[1]

        return idx, data_LR, data_HR

    def _parse_test_(self, serialized_example, mu_sig=None):
        '''
            this method parses TFRecords for the models to read in for testing.

            inputs:
                serialized_example - should only contain LR images (no HR ground truth images)
                mu_sig             - mean, sigma if known

            outputs:
                idx     - an array of indicies for each sample
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

        data_LR = tf.decode_raw(example['data_LR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]

        return idx, data_LR

    def set_mu_sig(self, data_path, batch_size):
        '''
            Compute mu, sigma for all channels of data
            inputs:
                data_path - (string) should be the path to the TRAINING DATA since mu and sigma are
                            calculated based on the trainind data regardless of if pretraining,
                            training, or testing.
                batch_size - number of samples to grab each interation. will be passed in directly
                             from pretrain, train, or test method by default.
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
            Compute mu, sigma for all channels of data
            inputs:
                data_path - (string) should be the path to the TRAINING DATA since mu and sigma are
                            calculated based on the trainind data regardless of if pretraining,
                            training, or testing.
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
