import os

from numpy.core.defchararray import asarray
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
import numpy as np
from glob import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import random
import pandas as pd
import sklearn.metrics

from .skeleton import load_model, load_encoder
from .resnet import Resnet101, ResnetSmall, Resnet18
from .autoencoder import AutoencoderSmall
from ..data_tool import parse_samples
from .callbacks import CSVLogger, ModelSaver


def plot_confusion(mdir, y_true, y_pred):
    # confusion matrix
    cm = sklearn.metrics.confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    np.savetxt(mdir + '/confusion_matrix.csv', cm)
    
   


def parse_train(serialized, append_latlon=False, tmax=None, autoencoder=False):
    examples = parse_samples(serialized)

    N = tf.cast(tf.shape(examples['H'])[0], tf.int64)
    H,W,C = examples['H'][0], examples['W'][0], examples['C'][0]
    T_max = tmax or examples['T_max'][0]

    patch1 = tf.io.decode_raw(examples['patch1'], tf.float32)
    patch1 = tf.reshape(patch1, (-1, H,W,C))
    patch2 = tf.io.decode_raw(examples['patch2'], tf.float32)
    patch2 = tf.reshape(patch2, (-1, H,W,C))

    labels = tf.one_hot(examples['label'] - 1, tf.cast(T_max, tf.int32))

    if append_latlon:
        lat_cos = tf.math.cos(2*np.pi * tf.linspace(examples['lat_start'], examples['lat_end'], H) / 180)
        lat_sin = tf.math.sin(2*np.pi * tf.linspace(examples['lat_start'], examples['lat_end'], H) / 180)

        lon_cos = tf.math.cos(2*np.pi * tf.linspace(examples['long_start'], examples['long_end'], W) / 360)
        lon_sin = tf.math.sin(2*np.pi * tf.linspace(examples['long_start'], examples['long_end'], W) / 360)

        lat_cos = tf.tile(tf.transpose(lat_cos[:,:,None,None], [1,0,2,3]), [1, 1, W, 1])
        lat_sin = tf.tile(tf.transpose(lat_sin[:,:,None,None], [1,0,2,3]), [1, 1, W, 1])

        lon_cos = tf.tile(tf.transpose(lon_cos[:,:,None,None], [1,2,0,3]), (1, H, 1, 1))
        lon_sin = tf.tile(tf.transpose(lon_sin[:,:,None,None], [1,2,0,3]), (1, H, 1, 1))

        patch1 = tf.concat([patch1, lat_cos, lat_sin, lon_cos, lon_sin], axis=-1)
        patch2 = tf.concat([patch2, lat_cos, lat_sin, lon_cos, lon_sin], axis=-1)

    if autoencoder:
        both = tf.concat([patch1, patch2], axis=0)
        X = {'img1': both}
        y = both
    else:
        X = {'img1': patch1, 'img2': patch2}
        y = labels

    return X, y


def make_train_ds(files, batch_size, n_shuffle=1000, compression_type='ZLIB', append_latlon=False, tmax=None, autoencoder=False):
    assert files

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=4 if n_shuffle else None, compression_type=compression_type)
    
    if n_shuffle:
        ds = ds.shuffle(n_shuffle)

    ds = ds.batch(1)
    ds = ds.map(lambda x: parse_train(x, append_latlon, tmax, autoencoder))
    ds = ds.unbatch()
    
    if not autoencoder:
        ds = ds.filter(lambda X, y: tf.math.reduce_sum(y) == 1)
    
    ds = ds.batch(batch_size)
    
    # not required anymore
    #ds = ds.map(lambda X,y,w: (tf.nest.map_structure(remap, X), y, w))

    return ds.prefetch(None)


class Train:

    def __init__(self):
        paths = glob('/data2/rplearn/rplearn_train_1979_1998.*.tfrecords')
        self.data_path_train = paths
        
        self.data_path_eval = glob('/data2/rplearn/rplearn_eval_2000_2005.*.tfrecords')
        self.val_freq = 3
        self.n_classes = 31
        self.is_autoencoder = True

        #self.resnet = ResnetSmall(shape=(160,160,2), n_classes=self.n_classes, output_logits=False, shortcut='projection')
        #resnet = Resnet18((160,160,2), 16, output_logits=False, shortcut='projection')
        #resnet = Resnet101((160,160,2), 16, output_logits=False)
        self.resnet = AutoencoderSmall(shape=(160,160,2), shortcut='projection')

        self.model_dir = Path('/data/final_rp_models')
        self.prefix = 'autoencoder'
        self.description = '''
        Auotencoder
        '''

        self.start_time = datetime.today()
        self.checkpoint_dir = self.model_dir / '{}_{}'.format(self.prefix, self.start_time.strftime('%Y-%m-%d_%H%M'))

        self.train_ds = None
        self.eval_ds = None


    def setup_dir(self):
        os.makedirs(self.checkpoint_dir)
        if self.description:
            with open(self.checkpoint_dir / 'description.txt', 'w') as f:
                f.write(self.description)

        self.resnet.summary()
        with open(self.checkpoint_dir / 'model_summary.txt', 'w') as f:
            self.resnet.summary(f)


    def setup_ds(self, tmax, batch_size=128):
        train_ds = make_train_ds(self.data_path_train, batch_size, n_shuffle=2000, tmax=tmax, autoencoder=self.is_autoencoder)
        
        self.small_train_ds = train_ds.take(2000)  # order is shuffled but these are always the same 2000 batches 
        
        if True:
            self.train_ds = train_ds 
        else:
            # ablation study
            self.train_ds = train_ds.take(9340)

        self.eval_ds = make_train_ds(self.data_path_eval, batch_size, n_shuffle=1, tmax=tmax, autoencoder=self.is_autoencoder)


    def train(self):
        self.setup_dir()
        self.setup_ds(tmax=self.n_classes)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics = 'categorical_accuracy'

        csv_logger = CSVLogger(str(self.checkpoint_dir / 'training.csv'), keys=['lr', 'loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'], append=True, separator=' ')
        saver = ModelSaver(self.checkpoint_dir)
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau('loss', min_delta=4e-2, min_lr=1e-5, patience=8)
        callbacks = [csv_logger]

        optimizer = tf.keras.optimizers.SGD(momentum=0.9, clipnorm=5.0)

        self.resnet.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )    

        self.resnet.model.optimizer.learning_rate.assign(1e-1)
        
        # pretraining
        self.resnet.model.fit(
            self.small_train_ds, 
            validation_data=self.eval_ds, 
            validation_freq=self.val_freq, 
            epochs=20, 
            callbacks=callbacks,
            verbose=1,
            initial_epoch=0
        )
        
        # training
        callbacks += [lr_reducer, saver]  # only activate now
        self.resnet.model.fit(
            self.train_ds, 
            validation_data=self.eval_ds, 
            validation_freq=self.val_freq, 
            epochs=60, 
            callbacks=callbacks,
            verbose=2,
            initial_epoch=0
        )

    def train_autoencoder(self):
        self.setup_dir()
        self.setup_ds(tmax=0)

        loss = tf.keras.losses.MeanSquaredError(),
        metrics = []

        csv_logger = CSVLogger(str(self.checkpoint_dir / 'training.csv'), keys=['lr', 'loss', 'val_loss'], append=True, separator=' ')
        saver = ModelSaver(self.checkpoint_dir)
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau('loss', min_delta=4e-2, min_lr=1e-5, patience=6)
        callbacks = [csv_logger, lr_reducer, saver]

        optimizer = tf.keras.optimizers.SGD(momentum=0.9, clipnorm=5.0)

        self.resnet.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )    

        self.resnet.model.optimizer.learning_rate.assign(1e-2)
        
        # training
        self.resnet.model.fit(
            self.train_ds.repeat(), 
            validation_data=self.eval_ds.repeat(), 
            validation_freq=self.val_freq, 
            epochs=60, 
            callbacks=callbacks,
            verbose=1,
            initial_epoch=0,
            steps_per_epoch=15000,
            validation_steps=1000,
        )

    
    def evaluate_single(self, dir, on_train=False):
        if not self.train_ds:
            self.setup_ds(tmax=self.n_classes, batch_size=256)
        
        model = load_model(dir)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(
            loss=loss,
            metrics='categorical_accuracy'
        )

        metrics = {
            'loss': tf.keras.metrics.CategoricalCrossentropy(),
            'accuracy': tf.keras.metrics.CategoricalAccuracy(),
            'top3-accuracy': tf.keras.metrics.TopKCategoricalAccuracy(3)
        }

        ds = self.train_ds if on_train else self.eval_ds
        y_true = []
        y_pred = []
        for X, y, _ in ds:
            preds = model(X, training=False)  # not the most efficient way, but the most comfortable one for sure
            y_true.append(y)
            y_pred.append(preds)

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        for name in metrics:
            metrics[name].update_state(y_true, y_pred)
            metrics[name] = metrics[name].result()

        # confusion matrix
        cm = sklearn.metrics.confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

        return y_true, y_pred, metrics, cm

    
    def evaluate_all(self, dir, on_train=False):
        def extract_epoch(path):
            return int(os.path.relpath(path, dir)[5:])

        # find dirs and sort by integers (not lexicographically)
        model_dirs = glob(dir + '/epoch*/')      
        model_dirs = sorted(model_dirs, key=extract_epoch)
        
        if not self.train_ds:
            self.n_classes = load_model(model_dirs[0]).layers[-1].output_shape[-1]  # make sure that we setup the ds correctly
            self.setup_ds(tmax=self.n_classes, batch_size=256)

        results = {}
        for mdir in model_dirs:
            print(f'evaluating {mdir}')
            epoch = extract_epoch(mdir)

            y_true, y_pred, metrics, cm = self.evaluate_single(mdir) 
            results[epoch] = {k: float(v) for k,v in metrics.items()}

            np.savetxt(mdir + '/confusion_matrix.csv', cm)

            metric_strings = {k: f"{v:.3f}" for k,v in results[epoch].items()}
            print(f'epoch {epoch}: {metric_strings}')


        df = pd.DataFrame.from_dict(results, orient='index')
        df.columns = ['loss', 'accuracy', 'top3-accuracy']
        df.index.name = 'epoch'
        df.to_csv(dir + '/evaluation.csv')


    def calc_loss(self, dir, on_train, loss_func, layer=-1):
        if dir:
            encoder = load_encoder(dir)
            inp = encoder.input
            out = encoder.layers[layer].output
            encoder = tf.keras.Model(inputs=inp, outputs=out)
        else:
            def denorm(x):
                y = x * [1.5794525e-1, 1.6044095e-1] + [8.821452e-4, 3.2483143e-4]
                y =  tf.math.sign(y) * tf.math.expm1(tf.math.abs(y)) / 0.2  # alpha=0.2
                return y
                #return y*[2.8568757e-5, 5.0819430e-5] + [1.9464334e-8, 2.0547947e-7]
            
            encoder = denorm

        img1 = tf.keras.Input(shape=[160,160,2], name='img1_inp')
        img2 = tf.keras.Input(shape=[160,160,2], name='img2_inp')

        r1 = encoder(img1)
        r2 = encoder(img2)

        #loss = loss_func(r1, r2)
        model = tf.keras.Model(inputs={'img1': img1, 'img2': img2}, outputs=[r1,r2])

        self.setup_ds(tmax=None)
        ds = self.train_ds if on_train else self.eval_ds

        samples = {}
        for X,y,weight in ds.take(1000):
            r1,r2 = model(X, training=False)
            losses = loss_func(r1, r2)
            labels = np.argmax(y, axis=1)

            for loss, label in zip(losses, labels):
                if label not in samples:
                    samples[label] = []
                samples[label].append(loss)

        # potentially unsafe (not guaranteed to be contained), but should be ok
        means = np.asarray([np.mean(samples[i]) for i in sorted(samples)])
        stds = np.asarray([np.std(samples[i]) for i in sorted(samples)])
        
        return means, stds


    def evaluate_loss(self, dir, layer, on_train=True):
        if layer < 0:
            encoder = load_encoder(dir)
            layer = len(encoder.layers) + layer

        def l1(r1, r2):
            diffs = tf.math.abs(r1 - r2)
            return tf.math.reduce_mean(diffs, axis=[1,2,3])

        def l2(r1, r2):
            sq_diffs = tf.math.squared_difference(r1, r2)
            return tf.math.reduce_mean(sq_diffs, axis=[1,2,3])

        def psnr(r1, r2):
            mse = l2(r1, r2)
            maximum = tf.math.reduce_max(tf.math.abs(tf.concat([r1, r2], 0)))
            psnr =  20. * tf.math.log(maximum) / tf.math.log(10.) - 10. * tf.math.log(mse) / tf.math.log(10.)
            return 50-psnr

        def ssim(r1, r2):
            minimum = tf.math.reduce_min(tf.concat([r1, r2], 0))
            maximum = tf.math.reduce_max(tf.concat([r1, r2], 0))
            return 1 - (1+tf.image.ssim(r1 - minimum, r2 - minimum, maximum - minimum)) / 2
            #return tf.math.reduce_mean(ssim, axis=[1,2,3])


        metrics = {
            'l1': l1,
            'l2': l2,
            'psnr': psnr,
            'ssim': ssim
        }

        def optimize_alpha(m, c):
            # C = sum_i=1^N (alpha*l_i - m_i)**2
            #   = sum_i=1^N alpha^2*l_i^2 - 2*alpha*l_i*m_i + m_i^2
            #   = (sum l_i^2)*alpha^2 - 2*(sum l_i*m_i)*alpha + (sum m_i^2)
            #
            # dC/dalpha = 2*(sum l_i^2)*alpha - 2*(sum l_i*m_i) = 0
            # =>  alpha = (sum l_i*m_i) / (sum l_i^2)
            N = c.shape[0]
            return np.sum(c[N//2:] * m[N//2:]) / np.sum(c[N//2:]**2)

        # content loss is always computed as l2
        content_loss_mean, content_loss_std = self.calc_loss(dir, on_train, l2, layer)
        
        loss_means = {}
        for name, metric in metrics.items():
            metric_mean, metric_std = self.calc_loss(None, on_train, metric, layer)
            loss_means[name] = metric_mean
            alpha = optimize_alpha(metric_mean, content_loss_mean)
            df = pd.DataFrame({
                'metric_mean': metric_mean, 
                'metric_std': metric_std, 
                'content_loss_mean': content_loss_mean, 
                'content_loss_std': content_loss_std,
                'alpha': [alpha] * len(metric_mean)
            })
            df.to_csv(dir + f'/layer{layer}_{name}_loss.csv')
        
        alpha = optimize_alpha(loss_means['l2'], content_loss_mean)
        print(f'alpha={alpha}')
        with open(dir + f'/layer{layer}_scale.txt', 'w') as f:
            f.write(str(alpha))

def main():
    Train().train_autoencoder()
    #Train().train()

    #dir = '/data/final_rp_models/rnet-small-23c_2021-09-09_1831'
    #Train().evaluate_all(_dir)
    #Train().evaluate_loss(_dir + '/epoch27', layer=196)
    #Train().evaluate_loss(_dir + '/epoch27', layer=148)

    """
    _dir = '/data/final_rp_models/rnet-small-abla-15c_2021-09-22_1623'
    Train().evaluate_all(_dir)
    Train().evaluate_loss(_dir + '/epoch26', layer=196)
    Train().evaluate_loss(_dir + '/epoch26', layer=148)


    _dir = '/data/final_rp_models/rnet-small-abla-23c_2021-10-02_1508'
    Train().evaluate_all(_dir)
    Train().evaluate_loss(_dir + '/epoch33', layer=196)
    Train().evaluate_loss(_dir + '/epoch33', layer=148)

    _dir = '/data/final_rp_models/rnet-small-abla-31c_2021-10-10_1702'
    Train().evaluate_all(_dir)
    Train().evaluate_loss(_dir + '/epoch33', layer=196)
    Train().evaluate_loss(_dir + '/epoch33', layer=148)
    """

if __name__ == '__main__':
    main()
