import numpy as np
import matplotlib.pyplot as plt
from PhIREGANs import *
sys.path.append('../utils')
from utils import *

path_prefix = ""
status_options = ['training'] # or 'pre-training' or 'testing'

#GENERAL SETTINGS. SOME OF THESE CAN BE CHANGED USING phiregans.set{instance_val}(value) once phiregans object is initialized
#-------------------------------------------------------------
epoch_shift = 0
num_epochs = 1
r = [2,5] #[5] FOR MR TO HR, or for solar
batch_size = 100

#WIND
#-------------------------------------------------------------
variable_to_SR = 'wind' #or 'solar'

train_path = path_prefix +  '' #insert appropriate file here

test_path = path_prefix + 'test_data/' + variable_to_SR + '_example_LR_validation.tfrecord'

val_path = "" #CCSM or 1d training set data.

model_path = 'models/lr-mr_5x_' + variable_to_SR + '_model/SRGAN'


#SOLAR
#-------------------------------------------------------------
'''
variable_to_SR = 'solar' #or 'wind'

train_path = path_prefix +  '' #insert appropriate file here

test_path = path_prefix + 'test_data/' + variable_to_SR + '_example_LR_validation.tfrecord'

val_path = "" #CCSM or 1d training set data.

model_path = 'model/models/lr-mr_10x_' + variable_to_SR + '_model/SRGAN'
'''

if __name__ == '__main__':

    phiregans = PhIREGANs(1, 1e-4, epoch_shift, d_type = variable_to_SR)

    for status in status_options:
        if status == 'pre-training':

            model_saved = phiregans.pretrain(r, train_path, test_path, model_path, batch_size = 100)

            print("The model is saved as: ", model_saved)

        elif status == 'training':

            model_saved = phiregans.train(r, train_path, test_path, model_path, batch_size = batch_size)

            print("The model is saved as: ", model_saved)

        elif status == 'testing':
            #self, r, train_path, val_path, model_path, surfRough_path = None, batch_size = 100

            sr_val = phiregans.test(r, train_path, val_path, model_path, batch_size = batch_size)

        else:
            print('Please enter a valid status. Status can be pretraining, training, or testing.')
