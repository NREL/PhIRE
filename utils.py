import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

def image_out(LR, SR, HR, file_name):
    assert LR.shape[0] == SR.shape[0], "LR and SR contain a different number of images"
    assert SR.shape[1] == HR.shape[1] and SR.shape[2] == HR.shape[2], "HR and SR are different shapes"
    for i in range(LR.shape[0]):

        plt.figure(figsize = (8,4))

        vmin_u = np.min(SR[i,:,:,0])
        vmax_u = np.max(SR[i,:,:,0])
        vmin_v = np.min(SR[i,:,:,1])
        vmax_v = np.max(SR[i,:,:,1])

        plt.subplot(231)
        plt.imshow(LR[i,:,:,0], vmin = vmin_u, vmax = vmax_u, cmap = 'viridis', origin = 'lower')
        plt.title("LR Input", fontsize = 9)
        plt.ylabel("u", fontsize = 9)
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(232)
        plt.imshow(SR[i,:,:,0], vmin = vmin_u, vmax = vmax_u, cmap = 'viridis', origin = 'lower')
        plt.title("GANs SR", fontsize = 9)
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(233)
        plt.imshow(HR[i,:,:,0], vmin = vmin_u, vmax = vmax_u, cmap = 'viridis', origin = 'lower')
        plt.title("Ground Truth", fontsize = 9)
        plt.xticks([], [])
        plt.yticks([], [])

        cax2 = plt.axes([0.82, 0.52, 0.009, 0.38])
        plt.colorbar(cax=cax2)
        cax2.tick_params(labelsize=8)
        cax2.set_ylabel("m/s", fontsize = 9)

        plt.subplot(234)
        plt.imshow(LR[i,:,:,1], vmin = vmin_v, vmax = vmax_v, cmap = 'viridis', origin = 'lower')
        plt.ylabel("v", fontsize = 9)
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(235)
        plt.imshow(SR[i,:,:,1], vmin = vmin_v, vmax = vmax_v, cmap = 'viridis', origin = 'lower')
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(236)
        plt.imshow(HR[i,:,:,1], vmin = vmin_v, vmax = vmax_v, cmap = 'viridis', origin = 'lower')
        plt.xticks([], [])
        plt.yticks([], [])

        cax = plt.axes([0.82, 0.1, 0.009, 0.38])
        plt.colorbar(cax=cax)
        cax.set_ylabel("m/s", labelpad = -1, fontsize = 9)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

        wspace = 0.1   # the amount of width reserved for blank space between subplots
        hspace = 0.1

        plt.subplots_adjust(wspace = wspace, hspace = hspace)

        if not os.path.exists('../data_out/'):
            os.makedirs('../data_out/')

        plt.savefig('../data_out/' + file_name + "_"+str(index)+".png", bbox_inches='tight')