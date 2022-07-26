''' @author: Karen Stengel
'''
from .PhIREGANs import *
from .rplearn.serialization import load_encoder
from glob import glob
from pathlib import Path


def main():
    
    run_name = 'rnet-small-23c'
    compression='ZLIB'
    infiles = sorted(glob('/data/sebastian/sr/sr_train_1979_1998.*.tfrecords'))
    checkpoint = '/data/sebastian/models/mse-20210901-111709/pretraining/generator-5'
    r = [2,2]

    encoder_path = Path('/data/sebastian/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27/')
    encoder_layer = 196
    loss_scale = float((encoder_path / f'layer{encoder_layer}_scale.txt').read_text())

    mu_sig = None  # data is already log1p- and z-normalized

    #######################################################

    def _load():
        enc = load_encoder(encoder_path, encoder_layer)
        enc.summary()
        return enc

    encoder = _load if encoder_path else None

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
    if False:
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

        gan.N_epochs = 5
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
