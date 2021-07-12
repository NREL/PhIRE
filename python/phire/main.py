''' @author: Karen Stengel
'''
from .PhIREGANs import *
from .encoder import load_encoder
from glob import glob

from tensorflow.python.util import module_wrapper
module_wrapper._PER_MODULE_WARNING_LIMIT = 0


def main():
    
    data_type = 'resnet-small-16c-2xdata-pre2'
    compression='ZLIB'
    data_path_train = sorted(glob('/data/stengel/HR/patches_train_1979_1990.*.tfrecords'))
    checkpoint = None
    r = [2,2]

    encoder_path = '/data/repr_models_HR/resnet-small-16c-2xdata-pre2_2021-07-07_1205/epoch22/'
    encoder_layer = -1
    loss_scale = 0.18

    #mu_sig = [[275.28983, 1.8918675e-08, 2.3001131e-07], [16.951859, 2.19138e-05, 4.490682e-05]]
    # log values:
    mu_sig = [[0.008315503, 0.0028762482], [0.5266841, 0.5418187]]

    ################### ###################################

    if encoder_path:
        encoder = lambda: load_encoder(encoder_path, encoder_layer)
    else:
        encoder = None

    gan = PhIREGANs(
        data_type=data_type, 
        mu_sig=mu_sig,
        print_every=50,
        N_epochs=1,
        save_every=1,
        compression=compression,
        alpha_content=loss_scale,
        encoder=encoder
    )
    
    # Pretraining
    if True:
        gan.N_epochs = 5
        checkpoint = gan.pretrain(
            r=r,
            data_path=data_path_train,
            model_path=checkpoint,
            batch_size=128
        )
        print(checkpoint)

    # Training
    if True:
        gan.N_epochs = 15
        gan.learning_rate = 1e-4
        gan.epoch_shift = 0
        checkpoint = gan.train(
            r=r,
            data_path=data_path_train,
            model_path=checkpoint,
            batch_size=64
        )

        gan.N_epochs = 15
        gan.learning_rate = 1e-5
        gan.epoch_shift = 15
        checkpoint = gan.train(
            r=r,
            data_path=data_path_train,
            model_path=checkpoint,
            batch_size=64
        )
        print(checkpoint)


if __name__ == '__main__':
    main()
