import numpy as np
import matplotlib.pyplot as plt

#dummy data

#set parameters
r = [2,5] #should be prime factorization of total SR amount ie 2*5 = 10x SR

#model paths
lr_wind_model_path = 'lr-mr-10x_wind_model/SRGAN'
mr_wind_model_path = 'mr-hr-5x_wind_model/SRGAN'

#pass paths into test (or train or pretrain)
