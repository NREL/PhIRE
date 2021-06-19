import tensorflow as tf
from .skeleton import PretextModel

class ResnetBase(PretextModel):

    def __init__(self, shape, n_classes, shortcut='padded', **kwargs):
        assert shortcut == 'padded' or shortcut == 'projection'
        self.shortcut_type = shortcut

        if 'regularizer' not in kwargs:
            kwargs['regularizer'] = tf.keras.regularizers.L2(1e-4)

        super(ResnetBase, self).__init__(shape, n_classes, **kwargs)

    def resblock_down(self, x, filters, name, downscale=False):
        strides = 2 if downscale else 1

        # residual:
        r = self.conv(x, filters, 3, f'{name}_conv1', strides = strides, use_bias=False)
        r = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(r)
        r = tf.keras.layers.Activation(self.activation, name=f'{name}_act1')(r)
        
        r = self.conv(r, filters, 3, f'{name}_conv2', use_bias=False)
        r = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(r)
        
        # skip connection:
        shortcut = self.shortcut(x, filters, downscale, name)
        y = tf.keras.layers.Add(name=f'{name}_skip')([shortcut,r])
        y = tf.keras.layers.Activation(self.activation, name=f'{name}_act2')(y)

        return y

    def shortcut(self, x, filters, downscale, name):
        strides = 2 if downscale else 1

        if x.shape[-1] != filters:
            if self.shortcut_type == 'projection':
                shortcut = self.conv(filters, 1, use_bias=False, strides=strides,  name=f'{name}_projection')(x)
            else:
                shortcut = tf.pad(x, [[0,0], [0,0], [0,0], [0, filters - x.shape[-1]]])[:, ::strides, ::strides, :]
        else:
            shortcut = x[:, ::strides, ::strides, :]

        return shortcut


class ResnetSmall(ResnetBase):

    def create_encoder(self, shape):
        inp = tf.keras.Input(shape=shape, name='inp')
    
        x = self.conv(inp, 16, 3, 'in_conv')
        x = tf.keras.layers.Activation(self.activation, name='in_conv_act')(x)

        n = 5
        for i in range(1,n+1):
            x = self.resblock_down(x, 16, f'block1_{i}')
        
        for i in range(1,n+1):
            x = self.resblock_down(x, 32, f'block2_{i}', downscale = (i==1))
        
        for i in range(1,n+1):
            x = self.resblock_down(x, 64, f'block3_{i}', downscale = (i==1))
        
        for i in range(1,n+1):
            x = self.resblock_down(x, 128, f'block4_{i}', downscale = (i==1))
        
        encoder = tf.keras.Model(inputs=inp, outputs=x, name='encoder')
        return encoder


class Resnet18(ResnetBase):

    def create_encoder(self, shape):
        inp = tf.keras.Input(shape=shape, name='inp')
    
        x = self.conv(inp, 64, 7, 'in_conv', strides=2)
        x = tf.keras.layers.Activation(self.activation, name='in_conv_act')(x)
        x = tf.keras.layers.MaxPool2D(3,2, padding='SAME', name='pool')(x)

        for i in range(1,3):
            x = self.resblock_down(x, 64, f'block1_{i}')
        
        for i in range(1,3):
            x = self.resblock_down(x, 128, f'block2_{i}', downscale=(i==1))

        for i in range(1,3):
            x = self.resblock_down(x, 256, f'block3_{i}', downscale=(i==1))

        for i in range(1,3):
            x = self.resblock_down(x, 512, f'block4_{i}', downscale=(i==1))

        encoder = tf.keras.Model(inputs=inp, outputs=x, name='encoder')
        return encoder
