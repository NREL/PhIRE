import numpy as np
import matplotlib.pyplot as plt
import sys
from PhIREGANs import *
sys.path.append('utils')
from utils import *

path_prefix = ''
status_options = ['training'] # or 'pre-training' or 'testing'

#GENERAL SETTINGS. SOME OF THESE CAN BE CHANGED USING phiregans.set{instance_val}(value) once phiregans object is initialized
#-------------------------------------------------------------
epoch_shift = 0
num_epochs = 1
r = [[2,5], [5]] #[5] FOR MR TO HR, or for solar
batch_size = 10

#WIND
#-------------------------------------------------------------
variable_to_SR = 'wind' #or 'solar'

train_path = path_prefix +  '' #insert appropriate file here

test_path = path_prefix + 'example_data/' + variable_to_SR + '_example_LR_validation.tfrecord'

val_path = "" #CCSM or 1d training set data.

model_path_lr = 'models/lr-mr_5x_' + variable_to_SR + '_model/SRGAN'
model_path_hr = 'models/mr-hr_10x_' + variable_to_SR + '_model/SRGAN'


#SOLAR
#-------------------------------------------------------------
'''
variable_to_SR = 'solar' #or 'wind'

train_path = path_prefix +  '' #insert appropriate file here

test_path = path_prefix + 'example_data/' + variable_to_SR + '_example_LR_validation.tfrecord'

val_path = "" #CCSM or 1d training set data.

model_path = 'model/models/lr-mr_10x_' + variable_to_SR + '_model/SRGAN'
'''

if __name__ == '__main__':

    phiregans = PhIREGANs(1, 1e-4, epoch_shift, d_type = variable_to_SR, mu_sig = [0,0.0])

    sr_val = phiregans.test(r[0], train_path, val_path, model_path_lr)

    sr_val_hr = phiregans.test(r[1], train_path, sr_val, model_path_hr, batch_size = batch_size)
