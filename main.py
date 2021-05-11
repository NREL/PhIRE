''' @author: Karen Stengel
'''
from PhIREGANs import *

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# ERA5 - temp,div,vort
#-------------------------------------------------------------
data_type = 'era5'
data_path = '/data/stengel_train_1980_1994.tfrecords'
checkpoint = None
r = [2,2]
#mu_sig = [[275.37314, 1.8918975e-08, 2.3000993e-07], [16.993176, 2.1903368e-05, 4.4884804e-05]]

# log values:
mu_sig = [0.0313213187067455, 0.010297026079633344, 0.003200743146941997], [0.6336947512733786, 0.5377000573137944, 0.5490888494254759]



from encoder import load_encoder

if __name__ == '__main__':

    gan = PhIREGANs(
        data_type=data_type, 
        mu_sig=mu_sig,
        perceptual_loss=False,
        print_every=400,
        N_epochs=50,
        save_every=5
    )
    
    # Pretraining
    gan.N_epochs = 100
    checkpoint = gan.pretrain(
        r=r,
        data_path=data_path,
        model_path=checkpoint,
        batch_size=32
    )

    

    # Training
    gan.N_epochs = 200
    checkpoint = gan.train(
        r=r,
        data_path=data_path,
        model_path=checkpoint,
        batch_size=32
    )

    # Inference
    gan.test(r=r,
        data_path=data_path,
        model_path=checkpoint,
        batch_size=32,
        save_every=200
    )