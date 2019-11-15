import numpy as np
import matplotlib.pyplot as plt
from PhIREGANs import *
sys.path.append('../utils')
from utils import *

#dummy data
lr = "test_data/wind_example_LR_validation.tfrecord"
mr = "test_data/wind_example_MR_validation.tfrecord"

#set parameters for wind
r_lr = [2,5] #should be prime factorization of total SR amount ie 2*5 = 10x SR
r_mr = [5]

#parameters for LR and MR are the same for solar:
#r_solar = [5]

#model paths
lr_wind_model_path = 'models/lr-mr-10x_wind_model/SRGAN'
mr_wind_model_path = 'models/mr-hr-5x_wind_model/SRGAN'

#pass paths into test (or train or pretrain)
mr_sr_wind = PhIRE_test(r_lr, lr_wind_model_path, lr)
hr_sr_wind = test(r_mr, mr_wind_model_path, mr_sr_wind)

# HR and LR numpy arrays for figure making
LR = np.load("test_data/LR_Wind_exampleImg.npy")
HR = np.load("test_data/HR_Wind_exampleImg.npy")

image_out(LR, hr_sr_wind, HR, file_name = 'example_wind_SR')
