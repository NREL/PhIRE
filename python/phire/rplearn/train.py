import tensorflow as tf
from glob import glob
from pathlib import Path
from datetime import datetime
import os

from .resnet import ResnetSmall, Resnet18
from .data_tool import parse_samples
from .callbacks import CSVLogger, ModelSaver


def parse_train(serialized):
    examples = parse_samples(serialized)

    H,W,C = examples['H'][0], examples['W'][0], examples['C'][0]
    T_max = examples['T_max'][0]

    patch1 = tf.io.decode_raw(examples['patch1'], tf.float32)
    patch1 = tf.reshape(patch1, (-1, H,W,C))
    patch2 = tf.io.decode_raw(examples['patch2'], tf.float32)
    patch2 = tf.reshape(patch2, (-1, H,W,C))

    labels = tf.one_hot(examples['label'], tf.cast(T_max, tf.int32))

    X = {'img1': patch1, 'img2': patch2}
    y = labels

    return X, y


def make_train_ds(files, batch_size, n_shuffle=1000, compression_type='ZLIB'):
    assert files

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=4, compression_type=compression_type)
    
    if n_shuffle:
        ds = ds.shuffle(n_shuffle)

    ds = ds.batch(batch_size)
    ds = ds.map(parse_train)
    return ds.prefetch(None)


def main():
    data_path_train = sorted(glob('/data2/stengel/HR/rplearn_train_1979_1990.*.tfrecords'))
    data_path_eval = sorted(glob('/data2/stengel/HR/rplearn_train_1979_1990.*.tfrecords'))

    resnet = Resnet18((160,160,2), 4*8)

    model_dir = Path('/data/repr_models_HR')
    prefix = 'test'
    description = '''
    Hello
    '''

    start_time = datetime.today()
    checkpoint_dir = model_dir / '{}_{}'.format(prefix, start_time.strftime('%Y-%m-%d_%H%M'))

    os.makedirs(checkpoint_dir)
    if description:
        with open(checkpoint_dir / 'description.txt', 'w') as f:
            f.write(description)

    resnet.summary()
    with open(checkpoint_dir / 'model_summary.txt', 'w') as f:
        resnet.summary(f)


    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = 'categorical_accuracy'

    csv_logger = CSVLogger(checkpoint_dir / 'training.csv', keys=['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'], append=False, separator=' ')
    saver = ModelSaver(checkpoint_dir)
    callbacks = [saver, csv_logger]

    resnet.model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics
    )

    train_ds = make_train_ds(data_path_train, 128)
    eval_ds = make_train_ds(data_path_eval, 128)

    resnet.model.optimizer.learning_rate.assign(1e-4)
    resnet.model.fit(train_ds, validation_data=eval_ds, validation_freq=5, epochs=1, callbacks=callbacks, verbose=1, initial_epoch=0)

    resnet.model.optimizer.learning_rate.assign(1e-3)
    resnet.model.fit(train_ds, validation_data=eval_ds, validation_freq=5, epochs=51, callbacks=callbacks, verbose=1, initial_epoch=1)

    resnet.model.optimizer.learning_rate.assign(1e-4)
    resnet.model.fit(train_ds, validation_data=eval_ds, validation_freq=5, epochs=66, callbacks=callbacks, verbose=1, initial_epoch=51)


if __name__ == '__main__':
    main()