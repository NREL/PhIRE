''' @author: Karen Stengel
'''
from .PhIREGANs import *
from .encoder import load_encoder
from glob import glob


def main():
    
    run_name = 'mse'
    compression='ZLIB'
    infiles = sorted(glob('/data/sebastian/sr/sr_train_1979_1998.*.tfrecords'))
    checkpoint = None
    r = [2,2]

    encoder_path = None
    encoder_layer = -1
    loss_scale = 1.0

    mu_sig = None  # data is already log1p- and z-normalized

    #######################################################

    if encoder_path:
        encoder = lambda: load_encoder(encoder_path, encoder_layer)
    else:
        encoder = None

    gan = PhIREGANs(
        data_type=run_name, 
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
            data_path=infiles,
            model_path=checkpoint,
            batch_size=64
        )
        print('pretraining done!')

    # Training
    if True:
        gan.N_epochs = 15
        gan.learning_rate = 1e-4
        gan.epoch_shift = 0
        checkpoint = gan.train(
            r=r,
            data_path=infiles,
            model_path=checkpoint,
            batch_size=64
        )

        gan.N_epochs = 15
        gan.learning_rate = 1e-5
        gan.epoch_shift = 15
        checkpoint = gan.train(
            r=r,
            data_path=infiles,
            model_path=checkpoint,
            batch_size=64
        )


if __name__ == '__main__':
    main()
