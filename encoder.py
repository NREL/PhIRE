import tensorflow as tf


def swish(x):
    return x*tf.nn.sigmoid(x)


def load_encoder(path, layer_idx=-1):
    with open(path + '/encoder.json', 'r') as f:
        encoder_org = tf.keras.models.model_from_json(f.read(), {'swish': swish})
    
    encoder = tf.keras.Model(encoder_org.input, encoder_org.layers[layer_idx].output, trainable=False)
    encoder.summary()
    encoder.load_weights(path + '/encoder_weights.hdf5', by_name=True)

    return encoder