import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import strftime, time
import sys
sys.path.append('../utils')
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

    DEFAULT_LOSS_TYPE = 'MSE'
    DEFAULT_DATA_TYPE = 'wind'

    def __init__(self, num_epochs = None, learn_rate = None, e_shift = None, save = None, print = None, mu_sig = None, loss_t = None, d_type = None):

        self.N_epochs = num_epochs if num_epochs is not None else 1000
        self.learning_rate = learn_rate if learn_rate is not None else 1e-5
        self.epoch_shift = e_shift if e_shift is not None else 0
        self.save_every = save if save is not None else 1
        self.print_every = print if print is not None else 1000
        self.mu_sig = mu_sig if mu_sig is not None else [0, 0.0]
        self.loss_type = loss_t if loss_t is not None else 'MSE'
        self.data_type = d_type if d_type is not None else 'wind'
        self.LR_data_shape = None

        if self.loss_type in ['AE', 'MSEandAE']:
            self.loss_model = '../autoencoder/model/latest2d/autoencoder'
            #loss_model = '../autoencoder/model/latest3d/autoencoder'
        elif self.loss_type in ['VGG', 'MSEandVGG']:
            self.loss_model = '../VGG19/model/vgg19'

        # Set various paths for where to save data
        self.now = strftime('%Y%m%d-%H%M%S')
        self.model_name = '/'.join(['model', self.now])
        self.layerviz_path = '/'.join(['layer_viz_imgs', self.now])
        self.log_path ='/'.join(['training_logs', self.now])
        #print("model name: ", model_name)
        self.test_data_path ='data_out/' + self.data_type + '/' + self.model_name

    def setDataType(self, dt):
        self.data_type = dt

    def setLoss_type(self, lt):
        self.loss_type = lt

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

    def pre_train(self, r, train_path, test_path, model_path, batch_size = 100):

        """Pretrain network (i.e., no adversarial component)."""
        print("model name: ", self.model_name)
        self.set_mu_sig(train_path, batch_size)
        scale = np.prod(r)

        print('Initializing network ...', end=' ')
        # Set high- and low-res data place holders. Make sure the sizes match the data
        x_LR = tf.placeholder(tf.float32, [None,  self.LR_data_shape[1],  self.LR_data_shape[2], self.LR_data_shape[3]])
        x_HR = tf.placeholder(tf.float32, [None, self.LR_data_shape[1]*scale,  self.LR_data_shape[2]*scale, self.LR_data_shape[3]])
        
        # Initialize network and set optimizer
        model = SRGAN(x_LR, x_HR, r=r, status='pre-training', loss_type= self.loss_type)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_train_op = optimizer.minimize(model.g_loss, var_list= model.g_variables)
        init = tf.global_variables_initializer()

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        print('Done.')

        print('Number of generator params: %d' %(count_params(model.g_variables)))

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

        # Create summary values for TensorBoard
        g_loss_iters_ph = tf.placeholder(tf.float32, shape=None)
        g_loss_iters = tf.summary.scalar('g_loss_vs_iters', g_loss_iters_ph)
        iter_summ = tf.summary.merge([g_loss_iters])

        g_loss_epoch_ph = tf.placeholder(tf.float32, shape=None)
        g_loss_epoch = tf.summary.scalar('g_loss_vs_epochs', g_loss_epoch_ph)
        epoch_summ = tf.summary.merge([g_loss_epoch])

        summ_train_writer = tf.summary.FileWriter(self.log_path+'-train', tf.get_default_graph())
        summ_test_writer  = tf.summary.FileWriter(self.log_path+'-test',  tf.get_default_graph())

        with tf.Session() as sess:
            print('Training network ...')

            sess.run(init)

            # Load previously trained network
            if self.epoch_shift > 0:
                print('Loading pre-trained network...', end=' ')
                g_saver.restore(sess, model_path)
                print('Done.')

            # Load perceptual loss network data if necessary
            if self.loss_type in ['AE', 'MSEandAE', 'VGG', 'MSEandVGG']:
                # Restore perceptual loss network, if necessary
                print('Loading perceptual loss network...', end=' ')
                var = tf.global_variables()
                if self.loss_type in ['AE', 'MSEandAE']:
                    loss_var = [var_ for var_ in var if 'encoder' in var_.name]
                elif self.loss_type in ['VGG', 'MSEandVGG']:
                    loss_var = [var_ for var_ in var if 'vgg19' in var_.name]
                saver = tf.train.Saver(var_list=loss_var)
                saver.restore(sess, self.loss_model)
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

                        #print(np.min(batch_SR), np.mean(batch_SR), np.max(batch_SR))
                        sess.run(g_train_op, feed_dict=feed_dict)
                        gl = sess.run(model.g_loss, feed_dict={x_HR: batch_HR, x_LR: batch_LR})

                        summ = sess.run(iter_summ, feed_dict={g_loss_iters_ph: gl})
                        summ_train_writer.add_summary(summ, iters)

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
                        print("line 180, ", type(self.mu_sig[1]), type(self.mu_sig[0]), np.amin(batch_LR), np.mean(batch_LR), np.amax(batch_LR))

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

                    # Write performance to TensorBoard
                    summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_train})
                    summ_train_writer.add_summary(summ, epoch)

                    summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_test})
                    summ_test_writer.add_summary(summ, epoch)

                    print('Epoch took %.2f seconds\n' %(time() - start_time))

            # Save model after training is completed
            model_dr = '/'.join([self.model_name, 'pretrain', 'SRGAN_pretrain'])
            if not os.path.exists(self.model_name):
                os.makedirs(self.model_name)
            g_saver.save(sess, model_dr)

        print('Done.')
        return model_dr

    def train(self, r, train_path, test_path, model_path, batch_size=100):

        """Train network using GANs. Only run this after model has been sufficiently pre-trained."""

        self.set_mu_sig(train_path, batch_size)
        print(self.loss_type, self.mu_sig, self.LR_data_shape)
        scale = np.prod(r)

        print('Initializing network ...', end=' ')
        tf.reset_default_graph()

        # Set high- and low-res data place holders. Make sure the sizes match the data
        x_LR = tf.placeholder(tf.float32, [None,  self.LR_data_shape[1],  self.LR_data_shape[2], self.LR_data_shape[3]])
        x_HR = tf.placeholder(tf.float32, [None, self.LR_data_shape[1]*scale,  self.LR_data_shape[2]*scale, self.LR_data_shape[3]])

        # Initialize network and set optimizer
        model = SRGAN(x_LR, x_HR, r=r, status='training', loss_type=self.loss_type)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_train_op = optimizer.minimize(model.g_loss, var_list=model.g_variables)
        d_train_op = optimizer.minimize(model.d_loss, var_list=model.d_variables)
        init = tf.global_variables_initializer()

        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        gd_saver = tf.train.Saver(var_list=(model.g_variables+model.d_variables), max_to_keep=10000)
        print('Done.')

        print('Number of generator params: %d' %(count_params(model.g_variables)))
        print('Number of discriminator params: %d' %(count_params(model.d_variables)))

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

        # Create summary values for TensorBoard
        g_loss_iters_ph = tf.placeholder(tf.float32, shape=None)
        d_loss_iters_ph = tf.placeholder(tf.float32, shape=None)
        g_loss_iters = tf.summary.scalar('g_loss_vs_iters', g_loss_iters_ph)
        d_loss_iters = tf.summary.scalar('d_loss_vs_iters', d_loss_iters_ph)
        iter_summ = tf.summary.merge([g_loss_iters, d_loss_iters])

        g_loss_epoch_ph = tf.placeholder(tf.float32, shape=None)
        d_loss_epoch_ph = tf.placeholder(tf.float32, shape=None)
        g_loss_epoch = tf.summary.scalar('g_loss_vs_epochs', g_loss_epoch_ph)
        d_loss_epoch = tf.summary.scalar('d_loss_vs_epochs', d_loss_epoch_ph)
        epoch_summ = tf.summary.merge([g_loss_epoch, d_loss_epoch])

        summ_train_writer = tf.summary.FileWriter(self.log_path+'-train', tf.get_default_graph())
        summ_test_writer  = tf.summary.FileWriter(self.log_path+'-test',  tf.get_default_graph())

        with tf.Session() as sess:
            print('Training network ...')

            sess.run(init)

            print('Loading pre-trained network...', end=' ')
            if 'SRGAN-all' in model_path:
                gd_saver.restore(sess, model_path) #continue training the descriminator if it exists.
            else:
                g_saver.restore(sess, model_path)

            print('Done.')

            # Load perceptual loss network data if necessary
            if self.loss_type in ['AE', 'MSEandAE', 'VGG', 'MSEandVGG']:
                # Restore perceptual loss network, if necessary
                print('Loading perceptual loss network...', end=' ')
                var = tf.global_variables()
                if self.loss_type in ['AE', 'MSEandAE']:
                    loss_var = [var_ for var_ in var if 'encoder' in var_.name]
                elif self.loss_type in ['VGG', 'MSEandVGG']:
                    loss_var = [var_ for var_ in var if 'vgg19' in var_.name]
                saver = tf.train.Saver(var_list=loss_var)
                saver.restore(sess, self.loss_model)
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
                            #print("generator training. discriminator loss : ", dl, "count : ", gen_count)
                            sess.run(g_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)
                            gen_count += 1

                        dis_count = 0
                        while (dl >= 0.6) and dis_count <30:
                            #generator fooled the discriminator. train the discriminator extra
                            #print("discriminator training. discriminator loss : ", dl, "count : ", dis_count)
                            sess.run(d_train_op, feed_dict=feed_dict)
                            gl, dl, p = sess.run([model.g_loss, model.d_loss, model.advers_perf], feed_dict=feed_dict)
                            dis_count += 1

                        pl, gal = sess.run([model.p_loss, model.g_ad_loss], feed_dict=feed_dict)
                        summ = sess.run(iter_summ, feed_dict={g_loss_iters_ph: gl, d_loss_iters_ph: dl})
                        summ_train_writer.add_summary(summ, iters)

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
                        batch_idx, batch_LR, batch_HR, batch_lat, batch_lon = None, None, None, None, None
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

                    # Write performance to TensorBoard
                    summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_train, d_loss_epoch_ph: d_loss_train})
                    summ_train_writer.add_summary(summ, epoch)

                    summ = sess.run(epoch_summ, feed_dict={g_loss_epoch_ph: g_loss_test, d_loss_epoch_ph: d_loss_test})
                    summ_test_writer.add_summary(summ, epoch)

                    print('Epoch took %.2f seconds\n' %(time() - start_time))
            g_model_dr, gd_model_dr = '/'.join([self.model_name, 'SRGAN', 'SRGAN']), '/'.join([self.model_name, 'SRGAN-all', 'SRGAN'])
            if not os.path.exists(self.model_name):
                os.makedirs(self.model_name)
            g_saver.save(sess, g_model_dr)
            gd_saver.save(sess, gd_model_dr)

        print('Done.')
        return [g_model_dr, gd_model_dr]

    def test(self, r, train_path, val_path, model_path, batch_size = 100):

        """Run test data through generator and save output."""
        self.set_mu_sig(train_path, batch_size)
        scale = np.prod(r)

        idx, LR_out, lat, lon = None, None, None, None
        model = None
        print('Initializing network ...', end=' ')
        tf.reset_default_graph()
        # Set low-res data place holders
        x_LR = tf.placeholder(tf.float32, [None, None, None, 2])

        model = None
        idx, LR_out, HR_out, lat, lon = None, None, None, None, None

        # Initialize network
        model = SRGAN(x_LR, r=r, status='testing', loss_type=self.loss_type)

        # Initialize network and set optimizer
        init = tf.global_variables_initializer()
        g_saver = tf.train.Saver(var_list=model.g_variables, max_to_keep=10000)
        print('Done.')

        print('Building data pipeline ...', end=' ')

        ds_test = tf.data.TFRecordDataset(val_path)
        iterator = None

        ds_test = ds_test.map(lambda xx: self._parse_val_(xx, self.mu_sig)).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(ds_test.output_types,
                                                   ds_test.output_shapes)
        idx, LR_out = iterator.get_next()

        print(idx)
        init_iter_test  = iterator.make_initializer(ds_test)
        print('Done.')

        # Set expected size of SR data.
        with tf.Session() as sess:
            print('Loading saved network ...', end=' ')
            sess.run(init)

            # Load trained model
            g_saver.restore(sess, model_path)
            print('Done.')

            print('Running test data ...')
            # Loop through test data
            sess.run(init_iter_test)
            try:
                data_out = None
                while True:

                    batch_idx, batch_LR,batch_lat, batch_lon = None, None, None, None, None, None
                    batch_idx, batch_LR, batch_HR = sess.run([idx, LR_out, HR_out])
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

            test_out_path = self.test_data_path

            if not os.path.exists(test_out_path):
                os.makedirs(test_out_path)
            if (epoch % self.save_every) == 0:
                np.save(test_out_path +'/val_SR.npy', data_out)

        print('Done.')
        return LR_out, data_out

    # Parser function for data pipeline. May need alternative parser for tfrecords without high-res counterpart
    def _parse_train_(self, serialized_example, mu_sig=None):
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

    def _parse_val_(self, serialized_example, mu_sig=None):
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
        # Compute mu, sigma for all channels of data
        # NOTE: This includes building a temporary data pipeline and looping through everything.
        #       There's probably a smarter way to do this...
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train_).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        _, LR_out, HR_out = iterator.get_next()
        #iter_out = iterator.get_next()
        #print(iter_out)
        with tf.Session() as sess:
            N, mu, sigma = 0, 0, 0
            try:
                while True:
                    data_HR = sess.run(HR_out)
                    data_LR = sess.run(LR_out)
                    #print(data_HR.shape, np.min(data_HR),np.mean(data_HR),np.max(data_HR))
                    N_batch, h, w, c = data_HR.shape
                    N_new = N + N_batch
                    mu_batch = np.mean(data_HR, axis=(0, 1, 2))
                    sigma_batch = np.var(data_HR, axis=(0, 1, 2))

                    sigma = (N/N_new)*sigma + (N_batch/N_new)*sigma_batch + (N*N_batch/N_new**2)*(mu - mu_batch)**2
                    mu = (N/N_new)*mu + (N_batch/N_new)*mu_batch

                    N = N_new

            except Exception as e:#tf.errors.OutOfRangeError:
                #print(e)
                #print("error")
                pass
        self.mu_sig = [mu, np.sqrt(sigma)]
        self.LR_data_shape = data_LR.shape
        print('Done.')
