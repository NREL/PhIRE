import tensorflow as tf

class MaskedLoss(tf.keras.losses.Loss):

        def __init__(self, loss, **kwargs):
            self.loss = loss
            super().__init__(**kwargs)

        def call(self, y_true, y_pred):
            y_true = y_true[:, 39:119, 39:119, :]
            y_pred = y_pred[:, 39:119, 39:119, :]

            # Cant use tf.shape here if we want this to work with ContentLoss
            # because keras can't properly figure out the graph dependency if
            # we use losses.Loss.
            # At this point you have to be madly stupid not to switch to torch...

            return self.loss.call(y_true, y_pred)


class ContentLoss(tf.keras.losses.Loss):

    def __init__(self, encoder, scale=1.0, **kwargs):
        self.encoder = encoder
        self.scale = scale

        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        enc_true = self.encoder(y_true, training=False)
        enc_pred = self.encoder(y_pred, training=False)

        enc_loss = tf.math.reduce_mean(tf.math.squared_difference(enc_true, enc_pred), axis=[1,2,3])
        mse_loss = tf.math.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=[1,2,3])

        return 0.9 * self.scale * enc_loss + 0.1 * mse_loss