import os
import sys
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhIREGANs import PhIREGANs


def denorm_log1p(x):
    return np.sign(x) * np.expm1(np.fabs(x))


def plot(img_LR, img_SR, img_HR, W,H, patch=None, log1p_denorm=False, cmap=None):
    if log1p_denorm:
        img_LR, img_SR, img_HR = denorm_log1p(img_LR), denorm_log1p(img_SR), denorm_log1p(img_HR)

    if patch:
        img_SR = img_SR[patch[0]:patch[2], patch[1]:patch[3], ...]
        img_HR = img_HR[patch[0]:patch[2], patch[1]:patch[3], ...]
        img_LR = img_LR[patch[0]//4:patch[2]//4, patch[1]//4:patch[3]//4, ...]

    min_, max_ = np.min(img_HR, axis=(0,1)), np.max(img_HR, axis=(0,1))
    n_channels=img_SR.shape[-1]

    plt.figure(figsize=(W*3, H*n_channels))
    for C in range(n_channels):
        plt.subplot(3,n_channels, 1 + C)
        plt.imshow(img_LR[...,C], vmin=min_[C], vmax=max_[C], cmap=cmap, interpolation=None)

        plt.subplot(3,n_channels, 1 + n_channels + C)
        plt.imshow(img_SR[...,C], vmin=min_[C], vmax=max_[C], cmap=cmap, interpolation=None)

        plt.subplot(3,n_channels, 1 + 2*n_channels + C)
        plt.imshow(img_HR[...,C], vmin=min_[C], vmax=max_[C], cmap=cmap, interpolation=None)

    return img_LR, img_SR, img_HR


def plot_dataset(gan, r, dataset, checkpoint, outdir, W,H, patch=None, log1p_denorm=False, cmap=None, batch_size=64, save_every=None):
    os.makedirs(outdir, exist_ok=True)

    iter_ = gan.test(r, dataset, checkpoint, batch_size=batch_size, save_every=save_every, return_batches=True, return_hr=True)

    images = []
    for i, (batch_LR, batch_SR, batch_HR) in enumerate(iter_):
        img_LR, img_SR, img_HR = batch_LR[0], batch_SR[0], batch_HR[0]
        img_LR, img_SR, img_HR = plot(img_LR, img_SR, img_HR, W,H, patch, log1p_denorm, cmap)

        images.append(img_SR)

        plt.savefig(outdir + '/{:03}.png'.format(i+1))
        plt.close()

    return images


def main():
    
    name = 'resnet'
    dataset = sorted(glob('/data2/stengel/whole_images/stengel_eval_1995_1999.*.tfrecords'))
    
    checkpoint = 'models/resnet-20210525-114614/pretraining/generator'
    epochs = range(5, 1, -1)

    if False:
        W,H = 7, 4
        patch = [0,0,256,512]
    else:
        W,H = 4, 4
        patch = [150,250,150+96,250+96]

    cmap = 'cividis'
    r = [2,2]
    mu_sig = [[0,0,0], [1,1,1]]
    log1p_denorm = False

    #############################################################

    gan = PhIREGANs(name, mu_sig=mu_sig, print_every=40)

    images = []
    for epoch in epochs:
        print('visualizing epoch ' + str(epoch))
        sys.stdout.flush()

        cp = '{}-{}'.format(checkpoint, epoch)
        figure_dir = 'figures_out/{}{}'.format(name, epoch)
        images_SR = plot_dataset(gan, r, dataset, cp, figure_dir,W,H,patch,log1p_denorm,cmap,save_every=40)
        images.append(images_SR[0])

    # progression
    n_channels = images[0].shape[-1]
    plt.figure(figsize=(W*n_channels, H*len(images)))
    min_, max_ = np.min(images, axis=(0,1,2)), np.max(images, axis=(0,1,2))
    for i, img in enumerate(images):
        for C in range(n_channels):
            plt.subplot(len(images),n_channels, i*n_channels + C + 1)
            plt.imshow(img[...,C], vmin=min_[C], vmax=max_[C], cmap=cmap, interpolation=None)

    plt.savefig(figure_dir + '/progression.png')


if __name__ == '__main__':
    main()
