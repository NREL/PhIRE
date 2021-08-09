import json
import numpy as np
import matplotlib.pyplot as plt

from .base import EvaluationMethod
from phire.utils import Welford


class Moments(EvaluationMethod):

    def __init__(self):
        super(Moments, self).__init__()
        self.welford_sr = Welford()
        self.welford_hr = Welford()

    
    def evaluate_both(self, idx, LR, SR, HR):
        self.welford_sr.update(SR, axis=0)
        self.welford_hr.update(HR, axis=0)


    def finalize(self):
        C = self.welford_sr.mean.shape[-1]

        mean_sr, std_sr = self.welford_sr.mean, self.welford_sr.std
        mean_hr, std_hr = self.welford_hr.mean, self.welford_hr.std

        # SR
        np.save(self.dir / 'mean.npy', mean_sr)
        np.save(self.dir / 'std.npy', std_sr)

        for c in range(C):
            plt.imsave(self.dir / f'mean_{c}.png', mean_sr[..., c])
            plt.imsave(self.dir / f'std_{c}.png', std_sr[..., c])

        # HR
        np.save(self.dir / 'mean_groundtruth.npy', mean_hr)
        np.save(self.dir / 'std_groundtruth.npy', std_hr)

        for c in range(C):
            plt.imsave(self.dir / f'mean_{c}_grountruth.png', mean_hr[..., c])
            plt.imsave(self.dir / f'std_{c}_groundtruth.png', std_hr[..., c])

        # Diff
        for c in range(C):
            plt.imsave(self.dir / f'mean_{c}_diff.png', mean_sr[..., c] - mean_hr[..., c])
            plt.imsave(self.dir / f'std_{c}_diff.png', std_sr[..., c] - std_hr[..., c]) 


    def summarize(self, paths, outdir):
        means = {k: np.load(path / 'mean.npy') for k,path in paths.items()}
        stds = {k: np.load(path / 'std.npy') for k,path in paths.items()}

        total_error = {k: np.sum((mean - means['groundtruth'])**2) for k, mean in means.items()}
        with open(outdir / 'cumulative_squared_error.json', 'w') as f:
            json.dump(total_error, f)