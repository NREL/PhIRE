import tensorflow as tf

def swish(x):
    return x*tf.nn.sigmoid(x)


def load_encoder(path):
    with open(path + '/encoder.json', 'r') as f:
        encoder_org = tf.keras.models.model_from_json(f.read(), {'swish': swish})
    
    encoder_org.load_weights(path + '/encoder_weights.hdf5')
    
    #encoder = tf.keras.Model(encoder_org.input, encoder_org.layers[-2].output, trainable=False)
    encoder = tf.keras.Model(encoder_org.input, encoder_org.layers[-1].output, trainable=False)
    encoder.summary()

    return encoder