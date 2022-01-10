import tensorflow as tf
import json
from pathlib import Path


def swish(x):
    return x*tf.nn.sigmoid(x)


def L2(l2=1e-2):
    return tf.keras.regularizers.l2(l2)


tf1_custom_objects = {'swish': swish, 'L2': L2}


def load_encoder(path, layer_idx=-1, input_shape=None):
    path = Path(path)
    with open(path / 'encoder.json', 'r') as f:
        encoder_org = tf.keras.models.model_from_json(f.read(), tf1_custom_objects)

    if input_shape:
        mdef = json.loads(encoder_org.to_json())
        mdef['config']['layers'][0]['config']['batch_input_shape'] = list((None,) + input_shape)

        enc_tmp = tf.keras.models.model_from_json(json.dumps(mdef), tf1_custom_objects)

        input_l = enc_tmp.input
        out_l = enc_tmp.layers[layer_idx].output
    else:
        input_l = encoder_org.input
        out_l = encoder_org.layers[layer_idx].output

    encoder = tf.keras.Model(input_l, out_l, trainable=False)
    encoder.summary()
    encoder.load_weights(str(path / 'encoder_weights.hdf5'), by_name=True)

    return encoder