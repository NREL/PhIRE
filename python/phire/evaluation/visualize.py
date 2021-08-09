import os
import numpy as np
import matplotlib.pyplot as plt
from .base import EvaluationMethod


class Visualize(EvaluationMethod):

    def __init__(self, yx=None, patch_size=None, stride=None):
        super(Visualize, self).__init__()

        self.n = 0
        self.yx=yx
        self.patch_size=patch_size
        self.stride = stride or 1

        self.no_groundtruth = True


    def truncate(self, img, lr=False):
        if self.yx is None or self.patch_size is None:
            return img
        else:
            y,x = self.yx
            h,w = self.patch_size
            if lr:
                y //= 4
                x //= 4
                h //= 4
                w //= 4

            return img[y:y+h, x:x+w]


    def evaluate_both(self, i, LR, SR, HR):
        for img_LR, img_SR, img_HR in zip(LR, SR, HR):
            self.n += 1
            if (self.n - 1) % self.stride != 0:
                continue

            img_LR = self.truncate(img_LR, lr=True)
            img_SR = self.truncate(img_SR)
            img_HR = self.truncate(img_HR)

            directory = f'{self.dir}/{self.n - 1}'
            os.makedirs(directory, exist_ok=True)

            vmin_div, vmax_div = np.min(img_HR[...,0]), np.max(img_HR[...,0])
            vmin_vort, vmax_vort = np.min(img_HR[...,1]), np.max(img_HR[...,1])

            plt.imsave(f'{directory}/div.png', np.hstack([img_SR[...,0], img_HR[...,0]]), vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/vort.png', np.hstack([img_SR[...,1], img_HR[...,1]]), vmin=vmin_vort, vmax=vmax_vort)
            
            plt.imsave(f'{directory}/SR_div.png', img_SR[...,0], vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/SR_vort.png', img_SR[...,1], vmin=vmin_vort, vmax=vmax_vort)

            plt.imsave(f'{directory}/HR_div.png', img_HR[...,0], vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/HR_vort.png', img_HR[...,1], vmin=vmin_vort, vmax=vmax_vort)

            plt.imsave(f'{directory}/LR_div.png', img_LR[...,0], vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/LR_vort.png', img_LR[...,1], vmin=vmin_vort, vmax=vmax_vort)