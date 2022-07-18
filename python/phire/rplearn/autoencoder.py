import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Concatenate
from .resnet import ResnetBase
from .skeleton import EncoderDecoderModel


class AutoencoderSmall(EncoderDecoderModel, ResnetBase):

    def build_encoder(self):
        skipcons = []

        inp = tf.keras.Input(shape=self.shape, name='inp')
        
        x = inp
        skipcons += [x]

        x = self.conv_bn_act(x, 16, 8, 'in', strides=2)
        skipcons += [x]

        x = MaxPool2D(3,2, padding='SAME', name='pool')(x)
        x = self.encoder_down_block(x, 16, skipcons, 'block1', downscale=False)
        
        x = self.encoder_down_block(x, 32, skipcons, 'block2')
        x = self.encoder_down_block(x, 64, skipcons, 'block3')
        x = self.encoder_down_block(x, 128, skipcons, 'block4')

        encoder = tf.keras.Model(inputs=inp, outputs=x, name='encoder')

        return encoder, skipcons


    def encoder_down_block(self, x, channels, skipcons, name, downscale=True):
        n = 6
        for i in range(1,n+1):
            x = self.resblock_down(x, channels, f'{name}_{i}', downscale = downscale and (i==1))
        skipcons += [x]

        return x


    def build_decoder(self, skipcons):
        H,W,C = self.shape[0] // 2**5, self.shape[1] // 2**5, 128
        inp = tf.keras.Input(shape=(H,W,C), name='inp')
    
        x = self.spatial_flatten(inp, [H,W,C])
        x = self.dense_bn_act(x, H*W, 'spatial_mixing')
        x = self.spatial_deflate(x, [H,W,C])
        x = self.conv_bn_act(x, C, 1, 'channel_mixing')

        x = self.resblock_up(x, 64, f'block4_upconv', upscale=True)

        x = self.decoder_up_block(x, 64, skipcons[-2], 'block3')
        x = self.decoder_up_block(x, 32, skipcons[-3], 'block2')
        x = self.decoder_up_block(x, 16, skipcons[-4], 'block1')

        x = self.resblock_up(x, 16, f'inp_up', upscale=True)
        x = self.conv(x, 2, 8, f'out_conv', kernel_initializer='glorot_normal')
        
        decoder = tf.keras.Model(inputs=inp, outputs=x, name='decoder')

        return decoder

    def decoder_up_block(self, x, channels, skipcon, name):
        n = 6
        for i in range(1,n+1):
            x = self.resblock_up(x, channels, f'{name}_{i}', upscale = (i==n))

        return x


    def build_model(self):
        self.encoder, skipcons = self.build_encoder()
        self.decoder = self.build_decoder(skipcons)

        img = tf.keras.Input(shape=self.shape, name='img')
        x = self.encoder(img)
        x = self.decoder(x)

        return tf.keras.Model(
            inputs=img,
            outputs=x,
            name=self.name
        )