''' @author: Karen Stengel
'''
from PhIREGANs import *
from encoder import load_encoder
from glob import glob

from tensorflow.python.util import module_wrapper
module_wrapper._PER_MODULE_WARNING_LIMIT = 0


def main():
    # ERA5 - temp,div,vort
    #-------------------------------------------------------------
    data_type = 'stacked-3days-1,4'
    data_path_train = sorted(glob('/data2/stengel/patches/patches_train_1980_1994.*.tfrecords'))
    data_path_test = sorted(glob('/data/stengel/whole_images/stengel_eval_1995_1999.*.tfrecords'))
    checkpoint = 'models/era5-mse-patches-20210512-200433/pretraining/generator-40'
    r = [2,2]

    encoder_path = '/data/repr_models/stacked_2021-05-15_2350/epoch37/'
    encoder_layer = -2
    loss_scale = 9.5

    #mu_sig = [[275.28983, 1.8918675e-08, 2.3001131e-07], [16.951859, 2.19138e-05, 4.490682e-05]]
    # log values:
    mu_sig = [[0.034322508, 0.01029128, 0.0031989873], [0.6344424, 0.53678083, 0.54819226]]

    ######################################################

    if encoder_path:
        encoder = lambda: load_encoder(encoder_path, encoder_layer)
    else:
        encoder = None

    gan = PhIREGANs(
        data_type=data_type, 
        mu_sig=mu_sig,
        print_every=500,
        N_epochs=1,
        save_every=1,
        alpha_content=loss_scale,
        encoder=encoder
    )
    
    # Pretraining
    if False:
        gan.N_epochs = 1
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
        gan.learning_rate = 1e-5
        gan.epoch_shift = 30
        checkpoint = gan.train(
            r=r,
            data_path=data_path_train,
            model_path=checkpoint,
            batch_size=64
        )
        print(checkpoint)

    # Inference
    if False:
        gan.test(r=r,
            data_path=data_path_test,
            model_path=checkpoint,
            batch_size=32,
            save_every=200
        )


if __name__ == '__main__':
    main()
