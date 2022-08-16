import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
tf.enable_eager_execution()

from phire.rplearn.autoencoder import AutoencoderSmall
from phire.rplearn.serialization import load_model
from phire.rplearn.data import make_autoencoder_ds, make_inpaint_ds

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# autoenc - layer 148 - 0.1 alpha
autoenc_inpaint = '/data/inpainting_models/inpaint_autoenc148_2022-08-14_1614/epoch1/'
autoenc_adam_inpaint = '/data/inpainting_models/inpaint_autoenc148_2022-08-14_1638/epoch13/'
autoenc_adam_lowlr_lowbatch_inpaint = '/data/inpainting_models/inpaint_autoenc148_2022-08-14_1732/epoch9/'  # bs = 32
autoenc_adam_lowlr_inpaint = '/data/inpainting_models/inpaint_autoenc148_2022-08-15_1256/epoch9/'  # bs = 128



model = load_model(autoenc_adam_lowlr_inpaint)  

ds = make_inpaint_ds(['/data/rplearn/rplearn_train_1979_1998.20.tfrecords'], 100, n_shuffle=1)
for X, y in ds:
    break


def plot(x, y):
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    bar1 = axs[0,0].imshow(x[..., 0], vmin=-6, vmax=6)
    bar2  =axs[0,1].imshow(x[..., 1], vmin=-6, vmax=6)
    
    axs[1,0].imshow(y[0, ..., 0], norm=bar1.norm)
    axs[1,1].imshow(y[0, ..., 1], norm=bar2.norm)
    
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes('bottom', size='5%', pad=0.1)
    plt.colorbar(bar1, cax=cax, orientation='horizontal')
    
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes('bottom', size='5%', pad=0.1)
    plt.colorbar(bar2, cax=cax, orientation='horizontal')
    

def plot_inpaint(idx):
    label = y[idx].numpy()
    pred = model(X[idx:idx+1]).numpy()
    a = label.copy()
    a[59:99, 59:99] = pred[0, 59:99, 59:99]
    
    
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    bar1 = axs[0,0].imshow(label[..., 0], vmin=-6, vmax=6)
    bar2  =axs[0,1].imshow(label[..., 1], vmin=-6, vmax=6)
    
    axs[1,0].imshow(a[..., 0], norm=bar1.norm)
    axs[1,1].imshow(a[..., 1], norm=bar2.norm)
    
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes('bottom', size='5%', pad=0.1)
    plt.colorbar(bar1, cax=cax, orientation='horizontal')
    
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes('bottom', size='5%', pad=0.1)
    plt.colorbar(bar2, cax=cax, orientation='horizontal')
    
def plot_example(batch, idx):
    plot(batch[idx], model(batch[idx:idx+1]))

for i in range(0, 32, 4):
    plot_inpaint(i)
    plt.savefig(f'{i}.png', bbox_inches='tight')