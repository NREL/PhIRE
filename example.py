import numpy as np
import matplotlib.pyplot as plt
from PhIREGANs import *

#dummy data
lr = "validation_LR_valid.tfrecord"
mr = "validation_MR_valid.tfrecord"

#set parameters
r_lr = [2,5] #should be prime factorization of total SR amount ie 2*5 = 10x SR
r_mr = [5]

#model paths
lr_wind_model_path = 'lr-mr-10x_wind_model/SRGAN'
mr_wind_model_path = 'mr-hr-5x_wind_model/SRGAN'

#pass paths into test (or train or pretrain)
mr_sr_wind = PhIRE_test(r_lr, lr_wind_model_path, lr)
#hr_sr_wind = test(r_mr, mr_wind_model_path, mr_sr_wind)
