import tensorflow as tf
from pathlib import Path
import os

from ..utils import json_to_tf1


class CommonLayers:
    def __init__(self, activation='relu', regularizer=None, **kwargs):
        self.activation = activation
        self.regularizer = regularizer
        super().__init__(**kwargs)


    def conv(self, x, filters, kernel, name, **kwargs):
        if 'padding' not in kwargs:
            kwargs['padding'] = 'SAME'
        
        if 'strides' not in kwargs:
            kwargs['strides'] = 1

        if 'kernel_initializer' not in kwargs:
            kwargs['kernel_initializer'] = 'he_normal'

        if 'kernel_regularizer' not in kwargs and self.regularizer:
            kwargs['kernel_regularizer'] = self.regularizer

        return tf.keras.layers.Conv2D(filters, kernel, name=name, **kwargs)(x)

    
    def dense(self, x, filters, name, activation=None, **kwargs):
        if 'kernel_initializer' not in kwargs:
            kwargs['kernel_initializer'] = 'he_normal'

        if 'kernel_regularizer' not in kwargs and self.regularizer:
            kwargs['kernel_regularizer'] = self.regularizer

        return tf.keras.layers.Dense(filters, activation=activation, name=name, **kwargs)(x)

    
    def bn(self, x, name):
        return tf.keras.layers.BatchNormalization(name=name, epsilon=1.001e-5)(x)


    def act(self, x, name):
        return tf.keras.layers.Activation(self.activation, name=name)(x)


    def conv_bn_act(self, x, filters, kernel, name, **kwargs):
        x = self.conv(x, filters, kernel, name + '_conv', **kwargs)
        x = self.bn(x, name+'_bn')
        x = self.act(x, name+'_act')
        return x


class BaseModel:
    def __init__(self, shape, name='model', **kwargs):
        self.shape = shape
        self.name = name
        super().__init__(**kwargs)

        self.model = self.build_model()
        self.model.wrapper = self

    def build_model(self):
        raise NotImplemented()

    def load_weights(self, dir):
        self.model.load_weights(str(Path(dir) / 'model_weights.hdf5'))


    def save(self, dir):
        dir = Path(dir)
        os.makedirs(dir)

        with open(dir / 'model.json', 'w') as f:
            f.write(json_to_tf1(self.model.to_json()))

        self.model.save_weights(str(dir / 'model_weights.hdf5'), save_format='h5')
   

    def summary(self, file=None):
        self.model.summary(print_fn=lambda x: print(x, file=file))    


class EncoderDecoderModel(BaseModel):

    def build_encoder(self):
        raise NotImplementedError()


    def save(self, dir):
        BaseModel.save(self, dir)

        with open(dir / 'encoder.json', 'w') as f:
            f.write(json_to_tf1(self.encoder.to_json()))

        self.encoder.save_weights(str(dir / 'encoder_weights.hdf5'), save_format='h5')
        

    def summary(self, file=None):
        self.encoder.summary(print_fn=lambda x: print(x, file=file))
        print('', file=file)
        self.model.summary(print_fn=lambda x: print(x, file=file))


class PretextModel(EncoderDecoderModel):

    def __init__(self, n_classes, output_logits=True, **kwargs):
        self.n_classes = n_classes
        self.output_logits = output_logits
        super().__init__(**kwargs)
        

    def build_model(self):
        self.encoder = self.build_encoder()
        return self.build_pretext_model(self.encoder)


    def build_pretext_model(self, encoder):
        img1 = tf.keras.Input(shape=self.shape, name='img1_inp')
        img2 = tf.keras.Input(shape=self.shape, name='img2_inp')

        r1 = encoder(img1)
        r2 = encoder(img2)
    
        pred = self.build_tail(r1, r2)        

        return tf.keras.Model(
            inputs={'img1': img1, 'img2': img2},
            outputs=pred,
            name=self.name
        )


    def build_tail(self, r1, r2):
        x = tf.keras.layers.Concatenate(name='stack')([r1, r2])
        
        x = self.conv_bn_act(x, 128, 3, 'combined1', strides=2)
        x = self.conv_bn_act(x, 128, 3, 'combined2')
        x = tf.keras.layers.Flatten(name='flatten')(x)
        
        pred = self.dense(x, self.n_classes, 'pred', None if self.output_logits else 'softmax', use_bias=False)
        return pred

  



def load_model(dir, custom_objects=None):
    with open(Path(dir) / 'model.json', 'r') as f:
        model = tf.keras.models.model_from_json(f.read(), custom_objects)

    model.load_weights(str(Path(dir) / 'model_weights.hdf5'), by_name=True)
    return model


def load_encoder(dir, custom_objects=None):
    with open(Path(dir) / 'encoder.json', 'r') as f:
        model = tf.keras.models.model_from_json(f.read(), custom_objects)

    model.load_weights(str(Path(dir) / 'encoder_weights.hdf5'), by_name=True)
    return model