import tensorflow as tf
from pathlib import Path
import os

from ..utils import json_to_tf1

class PretextModel:

    def __init__(self, shape, n_classes, output_logits=True, activation='swish', name='pretext_model'):
        self.activation = activation
        self.shape = shape
        self.n_classes = n_classes
        self.output_logits = output_logits
        self.name = name

        self.img1 = tf.keras.Input(shape=shape, name='img1_inp')
        self.img2 = tf.keras.Input(shape=shape, name='img2_inp')

        self.encoder = self.create_encoder(shape)

        self.r1 = self.encoder(self.img1)
        self.r2 = self.encoder(self.img2)
    
        x = tf.keras.layers.Concatenate(name='stack')([self.r1, self.r2])
        x = self.conv(x, 256, 1, 'final_conv')
        x = tf.keras.layers.Activation(self.activation, name='final_conv_act')(x)

        x = tf.keras.layers.Flatten(name='flatten')(x)
        pred = tf.keras.layers.Dense(
            self.n_classes,
            use_bias=True,
            activation=None if output_logits else 'softmax',
            name='pred'
        )(x)

        self.model = tf.keras.Model(
            inputs={'img1': self.img1, 'img2': self.img2},
            outputs=[pred],
            name=self.name
        )
        self.model.wrapper = self


    def load_weights(self, dir):
        self.model.load_weights(Path(dir) / 'model_weights.hdf5')


    def save(self, dir):
        dir = Path(dir)
        os.makedirs(dir)
    
        with open(dir / 'encoder.json', 'w') as f:
            f.write(json_to_tf1(self.encoder.to_json()))

        with open(dir / 'model.json', 'w') as f:
            f.write(json_to_tf1(self.model.to_json()))

        self.encoder.save_weights(dir / 'encoder_weights.hdf5', save_format='h5')
        self.model.save_weights(dir / 'model_weights.hdf5', save_format='h5')
   

    def summary(self, file=None):
        self.encoder.summary(print_fn=lambda x: print(x, file=file))
        print('', file=file)
        self.model.summary(print_fn=lambda x: print(x, file=file))    


    def conv(self, x, filters, kernel, name, **kwargs):
        if 'padding' not in kwargs:
            kwargs['padding'] = 'SAME'
        
        if 'strides' not in kwargs:
            kwargs['strides'] = 1

        if 'kernel_initializer' not in kwargs:
            kwargs['kernel_initializer'] = 'he_normal'

        return tf.keras.layers.Conv2D(filters, kernel, name=name, **kwargs)(x)