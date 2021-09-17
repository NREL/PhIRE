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
import scipy.optimize

from .skeleton import load_model, load_encoder
from .resnet import Resnet101, ResnetSmall, Resnet18
from ..data_tool import parse_samples
from .callbacks import CSVLogger, ModelSaver


def parse_train(serialized, append_latlon=False, discount=False, tmax=None):
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

    if discount:
        weights = tf.where(examples['label'] <= (T_max // 2), 1.9, 0.1)  # scale by 1.9 to get comparable losses
    else:
        weights = tf.broadcast_to(1.0, (N,))

    X = {'img1': patch1, 'img2': patch2}
    y = labels

    return X, y, weights


def make_train_ds(files, batch_size, n_shuffle=1000, compression_type='ZLIB', append_latlon=False, discount=False, tmax=None):
    assert files

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=3, compression_type=compression_type)
    
    if n_shuffle:
        ds = ds.shuffle(n_shuffle)

    ds = ds.batch(1)
    ds = ds.map(lambda x: parse_train(x, append_latlon, discount, tmax))
    ds = ds.unbatch()
    ds = ds.filter(lambda X, y, weight: tf.math.reduce_sum(y) == 1)
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
        self.n_classes = 23

        self.resnet = ResnetSmall((160,160,2), self.n_classes, output_logits=False, shortcut='projection')
        #resnet = Resnet18((160,160,2), 16, output_logits=False, shortcut='projection')
        #resnet = Resnet101((160,160,2), 16, output_logits=False)

        self.model_dir = Path('/data/final_models')
        self.prefix = 'rnet-small-23c'
        self.description = '''
        # Model:
        Our small resnet architecture, 4 blocks with filters [16,32,64,128] and 6 residual blocks in each.
        Starts with strided 8x8x16 conv and 3x3 max-pool (stride 2) as in resnet.

        tail consists of two 3x3x128 convs with BN

        l2-reg:     1e-4
        batch-size: 128
        activation: relu
        initializer: he-normal

        # Data:
        full-res 160x160 patches with 3d lookahead
        31 patches per image, 1979-1998 (20 years) -> 1.8 million patches (14.1k batches)
        (full dataset size is 42 patches -> 2.45 million patches)            
        eval on 2000-2005

        normalized with f(x) = sign(x) = ln(1 + 0.2*x)

        # Input vars:
        divergence (log1p), relative_vorticity (log1p)

        # Training:
        SGD with momentum=0.9
        lr gets reduced on plateau by one order of magnitude, starting with 1e-1

        pretraining with reduced dataset set size (2000 batches -> 256k) is neccessary and done for 20 epochs
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


    def setup_ds(self, tmax):
        train_ds = make_train_ds(self.data_path_train, 128, n_shuffle=2000, tmax=tmax)
        
        self.small_train_ds = train_ds.take(2000)  # order is shuffled but these are always the same 2000 batches 
        self.train_ds = train_ds #train_ds.take(2*5400)  # for comparability

        self.eval_ds = make_train_ds(self.data_path_eval, 128, n_shuffle=None, tmax=tmax)


    def train(self):
        self.setup_ds(tmax=self.n_classes)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics = 'categorical_accuracy'

        csv_logger = CSVLogger(self.checkpoint_dir / 'training.csv', keys=['lr', 'loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'], append=True, separator=' ')
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


    def run(self):
        print('TRAINING DATASET NOT COMPLETE, MOVE FROM /data TO /data2!')
        self.setup_dir()
        self.train()

    
    def evaluate_single(self, dir, on_train=False):
        if not self.train_ds:
            self.setup_ds(tmax=self.n_classes)
        
        model = load_model(dir)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(
            loss=loss,
            metrics='categorical_accuracy'
        )

        ds = self.train_ds if on_train else self.eval_ds
        return model.evaluate(ds, verbose=1)

    
    def evaluate_all(self, dir, on_train=False):
        if not self.train_ds:
            self.setup_ds(tmax=self.n_classes)

        def extract_epoch(path):
            return int(os.path.relpath(path, dir)[5:])

        # find dirs and sort by integers (not lexicographically)
        model_dirs = glob(dir + '/epoch*/')      
        model_dirs = sorted(model_dirs, key=extract_epoch)
        
        results = {}

        for mdir in model_dirs:
            print(f'evaluating {mdir}')
            epoch = extract_epoch(mdir)
            results[epoch] = self.evaluate_single(mdir)

        df = pd.DataFrame.from_dict(results, orient='index')
        df.columns = ['loss', 'accuracy']
        df.index.name = 'epoch'
        df.to_csv(dir + '/evaluation.csv')


    def calc_loss(self, dir, on_train, layer=-1):
        if dir:
            encoder = load_encoder(dir)
            inp = encoder.input
            out = encoder.layers[layer].output
            encoder = tf.keras.Model(inputs=inp, outputs=out)
        else:
            encoder = tf.identity

        img1 = tf.keras.Input(shape=[160,160,2], name='img1_inp')
        img2 = tf.keras.Input(shape=[160,160,2], name='img2_inp')

        r1 = encoder(img1)
        r2 = encoder(img2)

        sq_diffs = tf.math.squared_difference(r1, r2)
        l2 = tf.math.reduce_mean(sq_diffs, axis=[1,2,3])

        model = tf.keras.Model(inputs={'img1': img1, 'img2': img2}, outputs=l2)

        self.setup_ds(tmax=None)
        ds = self.train_ds if on_train else self.eval_ds

        samples = {}
        for X,y,weight in ds.take(2000):
            preds = model(X)
            labels = np.argmax(y, axis=1)

            for pred, label in zip(preds, labels):
                if label not in samples:
                    samples[label] = []
                samples[label].append(pred)

        # potentially unsafe (not guaranteed to be contained), but should be ok
        means = np.asarray([np.mean(samples[i]) for i in sorted(samples)])
        stds = np.asarray([np.std(samples[i]) for i in sorted(samples)])
        
        return means, stds


    def evaluate_loss(self, dir, layer, on_train=True):
        if layer < 0:
            encoder = load_encoder(dir)
            layer = len(encoder.layers) + layer

        mse_means, mse_stds = self.calc_loss(None, on_train, layer)
        layer_means, layers_stds = self.calc_loss(dir, on_train, layer)

        df = pd.DataFrame({'mse_mean': mse_means, 'mse_std': mse_stds, 'layer_mean': layer_means, 'layer_std': layers_stds})
        df.to_csv(dir + f'/layer{layer}_loss.csv')
        
        # C = sum_i=1^N (alpha*l_i - m_i)**2
        #   = sum_i=1^N alpha^2*l_i^2 - 2*alpha*l_i*m_i + m_i^2
        #   = (sum l_i^2)*alpha^2 - 2*(sum l_i*m_i)*alpha + (sum m_i^2)
        #
        # dC/dalpha = 2*(sum l_i^2)*alpha - 2*(sum l_i*m_i) = 0
        # =>  alpha = (sum l_i*m_i) / (sum l_i^2)
        N = layer_means.shape[0]
        alpha = np.sum(layer_means[N//2:] * mse_means[N//2:]) / np.sum(layer_means[N//2:]**2)
        print(f'alpha={alpha}')

        with open(dir + f'/layer{layer}_scale.txt', 'w') as f:
            f.write(str(alpha))

def main():
    #Train().run()
    #Train().evaluate_all('/data/final_rp_models/rnet-small-23c_2021-09-09_1831')
    Train().evaluate_loss('/data/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27', layer=196)
    Train().evaluate_loss('/data/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27', layer=148)


if __name__ == '__main__':
    main()