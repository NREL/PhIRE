import os
import numpy as np
import matplotlib.pyplot as plt
from .base import EvaluationMethod
import imageio
import pyshtools as pysh
import seaborn as sns


def format_lat(x, pos):
    if x < 640.0:
        deg = 90 - 90 * x / 640.0
        return f'{deg:.0f} °N'
    else:
        deg = 180 * x / 1280.0 - 90
        return f'{deg:.0f} °S'

def format_lon(x, pos):
    if x < 1280.0:
        deg = 180 * x / 1280.0
        return f'{deg:.0f} °E'
    else:
        deg = 360 - 360 * x / 2560.0
        return f'{deg:.0f} °W'

class HighpassCounter(EvaluationMethod):

    def __init__(self, highpass, threshold):
        super(HighpassCounter, self).__init__()

        self.highpass = highpass
        self.treshold = threshold

        self.needs_sh = True
        self.no_groundtruth = True

        self.counts = None


    def evaluate_both_sh(self, i, LR, SR, HR, SR_sh, HR_sh):
        if not self.highpass:
            return


        for sr, hr in zip(SR_sh, HR_sh):
            # highpass
            sr[:, :self.highpass, ...] = 0   
            hr[:, :self.highpass, ...] = 0      

            C = sr.shape[-1]
            for c in range(C):
                img_sr = pysh.expand.MakeGridDH(sr[..., c], sampling=2)
                img_hr = pysh.expand.MakeGridDH(hr[..., c], sampling=2)

                if self.counts is None:
                    self.counts = np.zeros(img_hr.shape + (C,), dtype='u8')

                diffs = np.fabs(img_sr - img_hr)
                #self.counts[..., c] += np.fabs(diffs / img_hr) >= self.treshold
                self.counts[..., c] += diffs >= self.treshold


    def finalize(self):
        for c in range(self.counts.shape[-1]):
            fig, ax = plt.subplots(figsize=(9,5))
            sns.heatmap(self.counts[..., c], ax=ax)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))
            ax.yaxis.set_major_locator(plt.MultipleLocator(20 * 1280/180))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
            ax.xaxis.set_major_locator(plt.MultipleLocator(40 * 1280/180))

            fig.savefig(f'{self.dir}/channel_{c}.png', bbox_inches = 'tight')
            plt.close(fig)

        return super().finalize()