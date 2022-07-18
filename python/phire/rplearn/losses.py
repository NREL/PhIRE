import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

class MaskedL2(MeanSquaredError):

        def call(self, y_true, y_pred):
            s = tf.shape(X)
            H,W = s[1], s[2]

            H_l = H//4 - 1
            H_r = 3 * (H//4) - 1
            W_l = W//4 - 1
            W_r = 3 * (H//4) - 1

            y_true = y_true[:, H_l:H_r, W_l:W_r, :]
            y_pred = y_pred[:, H_l:H_r, W_l:W_r, :]
            return super().call(y_true, y_pred)