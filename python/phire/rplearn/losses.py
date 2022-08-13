import tensorflow as tf

class MaskedLoss(tf.keras.losses.Loss):

        def __init__(self, loss, **kwargs):
            self.loss = loss
            super().__init__(**kwargs)

        def call(self, y_true, y_pred):
            l = 80 - 20 -1
            r = 80 + 20 -1

            y_true = y_true[:, l:r, l:r, :]
            y_pred = y_pred[:, l:r, l:r, :]

            # Cant use tf.shape here if we want this to work with ContentLoss
            # because keras can't properly figure out the graph dependency if
            # we use losses.Loss.
            # At this point you have to be madly stupid not to switch to torch...

            return self.loss.call(y_true, y_pred)


class ContentLoss(tf.keras.losses.Loss):

    def __init__(self, encoder, scale=1.0, **kwargs):
        self.encoder = encoder
        self.scale = scale
        self.mse = tf.keras.losses.MeanSquaredError()

        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        enc_true = self.encoder(y_true, training=False)
        enc_pred = self.encoder(y_pred, training=False)

        return self.scale * self.mse.call(enc_true, enc_pred)