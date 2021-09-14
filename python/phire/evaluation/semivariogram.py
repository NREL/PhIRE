import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyshtools as pysh

from .base import EvaluationMethod
from phire.utils import Welford


def _plot_semivariogram(lags, data):
    fig, ax = plt.subplots(figsize=(5.5,2.75))
    ax.set_yscale('log')
    ax.set_xscale('log')

    for name, (mean,std) in data.items():
        #ax.errorbar(lags, mean, std, label=name)
        ax.plot(lags, mean, label=name)

    ax.set_xlabel('lag distance')
    ax.legend()

    return fig, ax


class Semivariogram(EvaluationMethod):

    def __init__(self, lat_min = 0, lat_max = 180, long_min = 0, long_max=360, direction=None, n_samples=20):
        super(Semivariogram, self).__init__()
        
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.long_min = long_min
        self.long_max = long_max

        self.direction = direction
        self.n_samples = n_samples
        self.lags = np.asarray([25, 50, 100, 200, 300, 500, 1000])
        self.variances = Welford()


    def set_shape(self, shape):
        self.C = shape[-1]
        self.diffs = [[[] for _ in self.lags] for _ in range(self.C)]
    

    def _sample_3(self, img, radius):
        xyz = np.random.normal(size=(self.n_samples, 3))
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

        lats = 90 - 180*np.arccos(xyz[:,2])/np.pi  # [0, pi] -> [90, -90]
        lons = 180 + 180*np.arctan2(xyz[:,0], xyz[:,1])/np.pi  # [-pi, pi] -> [0, 360]

        y = self.sample_point(img, lats, lons) 
        
        if self.direction:
            directions = np.rint(359 * self.direction/2/np.pi)
        else:
            directions = np.random.randint(359, size=self.n_samples)

        angle = 360 * (radius / 6371) / 2 / np.pi
        lats2 = []
        lons2 = []
        for lat,lon,dir in zip(lats,lons, directions):
            p = pysh.utils.MakeCircleCoord(lat,lon,angle)[dir]
            lats2.append(p[0])
            lons2.append(p[1])

        y2 = self.sample_point(img, lats2, lons2)
        return y, y2

    def sample_point(self, img, lat, lon):
        H,W = img.shape[0], img.shape[1]
        lat = np.rint((90 - np.asarray(lat)) * (H-1)/180).astype('i8')
        lon = np.rint(np.asarray(lon) * (W-1)/360).astype('i8')

        return img[lat, lon]
    
    
    def evaluate_SR(self, idx, LR, SR):
        for img in SR:
            for c in range(self.C):
                for i in range(len(self.lags)):          
                    y1, y2 = self._sample_3(img[..., c], self.lags[i])
                    sq_diffs = (y1 - y2)**2
                    self.diffs[c][i] += list(sq_diffs)

        self.variances.update(SR, axis=(0,1,2))
    

    def finalize(self):
        diffs = np.asarray(self.diffs)  # C x lag x N
        means = np.mean(diffs, axis=2)  # SEMI-variogram adds factor 1/2 which we ignore since we normalize anyways
        stds = np.std(diffs, axis=2)

        np.savetxt(self.dir / 'semivariogram_mean.csv', means)
        np.savetxt(self.dir / 'semivariogram_std.csv', stds)
        np.savetxt(self.dir / 'lags.csv', list(self.lags))

        np.savetxt(self.dir / 'ds_mean.csv', self.variances.mean)
        np.savetxt(self.dir / 'ds_std.csv', self.variances.std)

        self.summarize({'SR': self.dir}, self.dir)


    def summarize(self, paths, outdir):
        means = []
        stds = []
        for path in paths.values():
            means.append(np.loadtxt(path / 'semivariogram_mean.csv'))
            stds.append(np.loadtxt(path / 'semivariogram_std.csv'))
            lags = np.loadtxt(path / 'lags.csv')

        if 'groundtruth' in paths:
            ds_var = np.loadtxt(paths['groundtruth'] / 'ds_std.csv')**2  # shape: (channel,)
        else:
            ds_var = np.loadtxt(list(paths.values())[0] / 'ds_std.csv')**2 


        means = np.stack(means, axis=0)  # run x channel x lag
        stds = np.stack(stds, axis=0)

        # normalize to variance of dataset:
        means = means / ds_var[None, :, None]
        stds = stds / ds_var[None, :, None]

        C = means.shape[1]
        for c in range(C):
            data = {name: (means[i,c], stds[i,c])for i, name in enumerate(paths)}
            with sns.plotting_context('paper'), sns.axes_style('whitegrid'), sns.color_palette('deep'):
                fig, ax = _plot_semivariogram(lags, data)
                fig.savefig(outdir / f'semivariogram_channel_{c}.png', bbox_inches='tight')
                fig.savefig(outdir / f'semivariogram_channel_{c}.pdf', bbox_inches='tight')
                fig.close()