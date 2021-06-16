import tensorflow as tf
from pathlib import Path


class ModelSaver(tf.keras.callbacks.Callback):

    def __init__(self, dir):
        self.dir = Path(dir)

    def on_epoch_end(self, epoch, logs):
        self.model.wrapper.save(self.dir / f'epoch{epoch}')

    def set_model(self, model):
        self.model = model


class CSVLogger(tf.keras.callbacks.CSVLogger):
    """
    Allows to write eval metrics with frequency != 1 by specifying keys manually
    """

    def __init__(self, filename, keys=None, **kwargs):
        super(CSVLogger, self).__init__(filename, **kwargs)
        self.keys = keys

    
    def on_epoch_end(self, epoch, logs):
        if self.keys:
            logs = logs or {}
            logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)
        
        super(CSVLogger, self).on_epoch_end(epoch, logs)