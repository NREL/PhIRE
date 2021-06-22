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
        r = self.bn(r, f'{name}_bn1')
        r = self.act(r, f'{name}_act1')
        
        r = self.conv(r, filters, 3, f'{name}_conv2', use_bias=False)
        r = self.bn(r, f'{name}_bn2')
        
        # skip connection:
        shortcut = self.shortcut(x, filters, downscale, name)
        y = tf.keras.layers.Add(name=f'{name}_skip')([shortcut,r])
        y = self.act(y, f'{name}_act2')

        return y


    def shortcut(self, x, filters, downscale, name):
        strides = 2 if downscale else 1

        if x.shape[-1] != filters:
            if self.shortcut_type == 'projection':
                shortcut = self.conv(x, filters, 1, use_bias=False, strides=strides,  name=f'{name}_projection')
            else:
                shortcut = tf.pad(x, [[0,0], [0,0], [0,0], [0, filters - x.shape[-1]]])[:, ::strides, ::strides, :]
        else:
            shortcut = tf.keras.layers.MaxPool2D(1, strides=strides, name=f'{name}_identity')(x)

        return shortcut


class ResnetSmall(ResnetBase):

    def create_encoder(self, shape):
        inp = tf.keras.Input(shape=shape, name='inp')
    
        x = self.conv_bn_act(inp, 16, 8, 'in', strides=2)
        x = tf.keras.layers.MaxPool2D(3,2, padding='SAME', name='pool')(x)

        n = 6
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
    
        x = self.conv_bn_act(inp, 64, 8, 'in', strides=2)
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


class Resnet101(PretextModel):

    def create_encoder(self, shape):
        inp = tf.keras.Input(shape=shape, name='inp')
        encoder = tf.keras.applications.resnet.ResNet101(include_top=False, weights=None, input_tensor=inp, pooling=None)
        return encoder