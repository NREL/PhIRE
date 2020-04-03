''' @author: Karen Stengel
'''
from PhIREGANs import *

# WIND - LR-MR
#-------------------------------------------------------------

data_type = 'wind'
train_path = 'example_data/wind_train_LR-MR.tfrecord'
test_path = 'example_data/wind_test_LR.tfrecord'
model_path = 'models/wind_lr-mr/trained_gan/gan'
r = [2, 5]
mu_sig=[[0.7684, -0.4575], [4.9491, 5.8441]]


# WIND - MR-HR
#-------------------------------------------------------------
'''
data_type = 'wind'
train_path = 'example_data/wind_train_MR-HR.tfrecord'
test_path = 'example_data/wind_test_MR.tfrecord'
model_path = 'models/wind_mr-hr/trained_gan/gan'
r = [5]
mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]]
'''

# SOLAR - LR-MR
#-------------------------------------------------------------
'''
data_type = 'solar'
train_path = 'example_data/solar_train_LR-MR.tfrecord'
test_path = 'example_data/solar_test_LR.tfrecord'
model_path = 'models/solar_lr-mr/trained_gan/gan'
r = [5]
mu_sig=[[344.3262, 113.7444], [370.8409, 111.1224]]
'''

# SOLAR - MR-HR
#-------------------------------------------------------------
'''
data_type = 'solar'
train_path = 'example_data/solar_train_MR-HR.tfrecord'
test_path = 'example_data/solar_test_MR.tfrecord'
model_path = 'models/solar_mr-hr/trained_gan/gan'
r = [5]
mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]
'''

if __name__ == '__main__':

    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
    
    model_dir = phiregans.pretrain(r=r,
                                   data_path=train_path,
                                   model_path=model_path,
                                   batch_size=1)

    model_dir = phiregans.train(r=r,
                                data_path=train_path,
                                model_path=model_dir,
                                batch_size=1)

    phiregans.test(r=r,
                   data_path=test_path,
                   model_path=model_dir,
                   batch_size=1,
                   plot_data=True)


