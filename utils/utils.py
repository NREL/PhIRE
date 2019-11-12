import numpy as np
import matplotlib.pyplot as plt
from time import time

def load_data(path=None, N_test=None, seed=11):
    """Load dataset and partition into training and testing data."""

    if N_test is None:
        # Separate files for train/test data
        train_x = np.load(path[0])
        test_x  = np.load(path[1])

    else:
        # Single file of data that needs to be partitioned
        data_x = np.load(path)

        N = data_x.shape[0]
        N_train = N - N_test

        np.random.seed(seed)
        indices = np.random.permutation(N)

        np.random.seed(int(time()))

        train_x, test_x = data_x[indices[:N_train]], data_x[indices[N_train:]]

    return train_x, test_x


def count_params(var_list):
    """Count the number of trainable parameters in var_list."""
    num_vars = 0
    for var in var_list:
        tmp = 1
        for n in list(var.shape):
            tmp *= n
        num_vars += tmp
    return num_vars


def write_file(file, txt, mode='a'):
    """Write data to file."""
    fid = open(file, mode)
    fid.write(txt)
    fid.close()


def image_to_grid(x):
    """Transform data across multiple channels into a single gridded image"""
    """Adapated from tf_cnnvis package"""
    if x.ndim == 4:
        h, w, d, C = x.shape
        x = x[:, :, int(d/2), :]
    elif x.ndim== 3:
        h, w, C = x.shape
    else:
        print('Error')
        exit()

    grid_size = int(np.ceil(np.sqrt(C)))

    padding = 1
    grid_height = h*grid_size + padding*(grid_size - 1)
    grid_width  = w*grid_size + padding*(grid_size - 1)
    grid = np.zeros((grid_height, grid_width))
    grid.fill(np.nan)

    next_idx = 0
    y0, y1 = 0, h
    for i in range(grid_size):
        x0, x1 = 0, w
        for j in range(grid_size):
            if next_idx < C:
                grid[y0:y1, x0:x1] = x[:, :, next_idx]
                next_idx += 1
            x0 += (w + padding)
            x1 += (w + padding)
        y0 += (h + padding)
        y1 += (h + padding)

    return grid

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

        plt.savefig("../" + file_name + "_"+str(index)+".png", bbox_inches='tight')


class BatchGenerator:
    '''Generator class returning list of indexes at every iteration.'''
    def __init__(self, batch_size, dataset_size):
        self.batch_size   = batch_size
        self.dataset_size = dataset_size

        assert (self.dataset_size > 0)               , 'Dataset is empty.'
        assert (self.dataset_size >= self.batch_size), 'Invalid batch_size.'
        assert (self.batch_size > 0)                 , 'Invalid batch_size.'

        self.last_idx = -1
        self.idxs     = np.random.permutation(dataset_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_idx + self.batch_size <= self.dataset_size - 1:
            start = self.last_idx + 1
            self.last_idx += self.batch_size

            return self.idxs[start: self.last_idx + 1]

        else:
            if self.last_idx == self.dataset_size - 1:
                raise StopIteration

            start = self.last_idx + 1
            self.last_idx = self.dataset_size - 1

            return self.idxs[start: self.last_idx + 1]
