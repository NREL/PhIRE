import tensorflow as tf

class MaskedLoss(tf.keras.losses.Loss):

        def __init__(self, loss, **kwargs):
            self.loss = loss
            super().__init__(**kwargs)

        def call(self, y_true, y_pred):
            s = tf.shape(y_true)
            H,W = s[1], s[2]

            H_l = H//4 - 1
            H_r = 3 * (H//4) - 1
            W_l = W//4 - 1
            W_r = 3 * (H//4) - 1

            y_true = y_true[:, H_l:H_r, W_l:W_r, :]
            y_pred = y_pred[:, H_l:H_r, W_l:W_r, :]
            return self.loss(y_true, y_pred)