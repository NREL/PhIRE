import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from glob import glob
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import pyshtools as pysh
from spharm import Spharmt

from .PhIREGANs import PhIREGANs
from .utils import Welford


class EvaluationMethod:

    def __init__(self):
        self.no_groundtruth=False
        self.dir = None
        self.shape = None
        self.needs_sh = False


    def set_dir(self, dir):
        self.dir = dir
        

    def set_shape(self, shape):
        self.shape = shape


    def evaluate_both(self, i, LR, SR, HR):
        pass


    def evaluate_SR(self, i, LR, SR):
        pass

    
    def evaluate_SR_sh(self, i, LR, SR, SR_sh):
        pass


    def finalize(self):
        pass


    def summarize(self, paths, outdir):
        pass


class PowerSpectrum(EvaluationMethod):

    def __init__(self, H,W):
        super(PowerSpectrum, self).__init__()
        self.spharm = Spharmt(W,H, legfunc='stored')
        self.spectrums = []

        self.needs_sh = True
        

    def _calc_energy_spectrum(self, div_vort):
        sh = self.spharm.grdtospec(div_vort)
        u,v = self.spharm.getuv(sh[..., 1], sh[..., 0])
        
        u_sh = pysh.expand.SHExpandDH(u)
        u_spectrum = pysh.spectralanalysis.spectrum(u_sh)  # <-> u^2

        v_sh = pysh.expand.SHExpandDH(v)
        v_spectrum = pysh.spectralanalysis.spectrum(v_sh)  # <-> v^2

        return 0.5 * (u_spectrum + v_spectrum)  # <-> 0.5 * (u^2 + v^2)


    def set_shape(self, shape):
        self.shape = shape
        self.spectrums = [[] for _ in range(self.shape[-1])]
        self.energy_spectrum = []


    def evaluate_SR_sh(self, i, LR, SR, SR_sh):
        for c in range(SR_sh.shape[-1]):
            self.spectrums[c] += [pysh.spectralanalysis.spectrum(sh) for sh in SR_sh[..., c]]

        self.energy_spectrum += [self._calc_energy_spectrum(img) for img in SR]


    def finalize(self):
        spectrums = np.asarray(self.spectrums)  # C x N x L
        mean_spectrum = np.mean(spectrums, axis=1)

        for c, spectrum in enumerate(mean_spectrum):
            np.savetxt(self.dir / f'spectrum_channel_{c}.csv', spectrum)

        energy_spectrum = np.mean(np.asarray(self.energy_spectrum), axis=0)
        np.savetxt(self.dir / 'energy_spectrum.csv', energy_spectrum)

        self.summarize({'SR': self.dir}, self.dir)


    def summarize(self, paths, outdir):
        for c in range(self.shape[-1]):
            plt.figure(figsize=(7,4))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('wavenumber')
            plt.ylabel('energy')

            for name, path in paths.items():
                spectrum = np.loadtxt(path / f'spectrum_channel_{c}.csv')
                x = np.arange(1, 1+spectrum.shape[0])
                plt.plot(x, spectrum, label=name)

            plt.plot(x, x**(-5/3) * spectrum[0], '--')

            plt.savefig(outdir / f'spectrum_channel_{c}.png')

        plt.figure(figsize=(7,4))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('wavenumber')
        plt.ylabel('energy')

        for name, path in paths.items():
            spectrum = np.loadtxt(path / 'energy_spectrum.csv')
            x = np.arange(1, 1+spectrum.shape[0])
            plt.plot(x, spectrum, label=name)

        plt.plot(x, x**(-5/3) * spectrum[0], '--')

        plt.savefig(outdir / 'energy_spectrum.png')


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



class Semivariogram(EvaluationMethod):

    def __init__(self, lat_min = 0, lat_max = 180, long_min = 0, long_max=360, direction=None, n_samples=20):
        super(Semivariogram, self).__init__()
        self.needs_sh = True
        
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.long_min = long_min
        self.long_max = long_max

        self.direction = direction
        self.n_samples = n_samples
        self.lags = np.asarray([10, 20, 30, 40, 50, 75, 100, 250, 500])
        self.variances = Welford()


    def set_shape(self, shape):
        self.C = shape[-1]
        self.diffs = [[[] for _ in self.lags] for _ in range(self.C)]


    def _sample(self, sh, r_degree):
        lat = np.random.uniform(-90, 90, size=self.n_samples)
        lon = np.random.uniform(0, 360, size=self.n_samples)

        if self.direction:
            directions = np.tile(self.direction, (self.n_samples, 1))
        else:
            directions = np.random.normal(size=2*self.n_samples).reshape((self.n_samples, 2))

        directions = r_degree * (directions / np.linalg.norm(directions, axis=1)[:, None])
        
        lat2 = lat+directions[:,0]
        lon2 = lon+directions[:,1]

        # wrap-around
        lon2[(lat2 > 90) | (lat2 < -90)] = lon2[(lat2 > 90) | (lat2 < -90)] + 180
        lat2[lat2 > 90] = 180 - lat2[lat2 > 90]
        lat2[lat2 < -90] = -180 - lat2[lat2 < -90]

        y1 = pysh.expand.MakeGridPoint(sh, lat=lat, lon=lon) 
        y2 = pysh.expand.MakeGridPoint(sh, lat=lat2, lon=lon2)

        return y1, y2


    def _sample_multiple(self, sh, radi):
        lat = np.random.uniform(-90, 90, size=(self.n_samples, len(radi)))
        lon = np.random.uniform(0, 360, size=(self.n_samples, len(radi)))

        if self.direction:
            directions = np.broadcast_to(self.direction[None, :, None], (self.n_samples, 2, len(radi)))
        else:
            directions = np.random.normal(size=self.n_samples*2*len(radi)).reshape((self.n_samples, 2, len(radi)))

        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        lat2 = lat + directions[:,0,:] * radi
        lon2 = lon + directions[:,1,:] * radi

        # wrap-around
        lon2[(lat2 > 90) | (lat2 < -90)] = lon2[(lat2 > 90) | (lat2 < -90)] + 180
        lat2[lat2 > 90] = 180 - lat2[lat2 > 90]
        lat2[lat2 < -90] = -180 - lat2[lat2 < -90]

        y = pysh.expand.MakeGridPoint(sh, lat=np.concatenate([lat, lat2]), lon=np.concatenate([lon, lon2])) 
        y = y.reshape(2, self.n_samples, -1)

        return y[0], y[1]


    def _sample_2(self, sh, radius):
        xyz = np.random.normal(size=(self.n_samples, 3))
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

        lats = 90 - 180*np.arccos(xyz[:,2])/np.pi  # [0, pi] -> [90, -90]
        lons = 180 + 180*np.arctan2(xyz[:,0], xyz[:,1])/np.pi  # [-pi, pi] -> [0, 360]

        y = pysh.expand.MakeGridPoint(sh, lats, lons) 
        
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

        y2 = pysh.expand.MakeGridPoint(sh, lats2, lons2)
        return y, y2

    def evaluate_SR_sh(self, i, LR, SR, SR_sh):
        for img_sh in SR_sh:
            for c in range(self.C):
                for i in range(len(self.lags)):          
                    y1, y2 = self._sample_2(img_sh[..., c], self.lags[i])
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
            plt.figure(figsize=(7,3))
            plt.yscale('log')
            plt.xscale('log')

            for i, name in enumerate(paths):
                plt.errorbar(lags, means[i,c], stds[i,c], label=name)

            plt.xlabel('lag distance')

            plt.savefig(self.dir / f'semivariogram_channel_{c}.png')


class Evaluation:

    def __init__(self):
        self.dir = Path('/data/results/srgan/resnet-small-16c/gan17/')
        self.checkpoint = '/data/results/models/resnet-small-16c-20210622-172045/training/gan-17'

        #self.dir = Path('/data/results/srgan/mse/gan17/')
        #self.checkpoint = '/data/results/models/mse-20210608-094931/training/gan-17'

        self.groundtruth = False
        self.denorm = True

        self.dataset = sorted(glob('/data/stengel/HR/sr_eval_2000_2002.*.tfrecords'))
        
        ##########################################

        self.mean_log1p, self.std_log1p = [0.008315503, 0.0028762482], [0.5266841, 0.5418187]
        self.mean, self.std = [2.0152406e-08, 2.1581373e-07], [2.8560082e-05, 5.0738556e-05]

        self.save_every = 1
        self.batch_size = 2  # cdnn limits to 2 giga-elements, 2x1280x2560x256 ~ 1.6giga
        self.r = [2,2]

        self.measure_time = False

        self.setup_metrics()


    def setup_metrics(self):
        self.px_per_deg = 2560 / 360
        img_patch_size = (self.to_px(60), self.to_px(60))
        img_freq = 20

        vis_sea = Visualize((self.to_px(60), self.to_px(90)), img_patch_size, img_freq)
        vis_eu = Visualize((self.to_px(40), self.to_px(0)), img_patch_size, img_freq)
        vis_na = Visualize((self.to_px(40), self.to_px(250)), img_patch_size, img_freq)
        vis_pac = Visualize((self.to_px(60), self.to_px(180)), img_patch_size, img_freq)

        if self.denorm:
            metrics = {
                'power-spectrum': PowerSpectrum(1280, 2560),
                'semivariogram': Semivariogram(n_samples=20),
                'img-SEA': vis_sea,
                'img-EU': vis_eu,
                'img-NA': vis_na,
                'img-pacific': vis_pac,
            }
        else:
            metrics = {
                'img-SEA-transformed': vis_sea,
                'img-EU-transformed': vis_eu,
                'img-NA-transformed': vis_na,
                'img-pacific-transformed': vis_pac,
            }

        self.metrics = {k:metric for k, metric in metrics.items() if not metric.no_groundtruth or not self.groundtruth}


    def to_px(self, deg):
        return int(round(deg*self.px_per_deg))


    def deprocess(self, batch):
        y = batch * self.std_log1p + self.mean_log1p
        y =  np.sign(y) * np.expm1(np.fabs(y))
        return y*self.std + self.mean


    def create_dirs(self):
        for name, metric in self.metrics.items():
            try:
                os.makedirs(self.dir / name)
            except OSError:
                resp = input(f'{self.dir / name} already exists. Continue? (y/n)')
                if resp != 'y' and resp != 'Y':
                    print('Evaluation cancelled')
                    return False
                else:
                    os.makedirs(self.dir / name, exist_ok=True)
            
            metric.set_dir(self.dir / name)
        
        return True


    def run(self):
        calc_sh = any(metric.needs_sh for metric in self.metrics.values())

        if not self.create_dirs():
            return

        gan = PhIREGANs('eval', mu_sig=[[0,0], [1,1]], print_every=1e9, compression='ZLIB')
        iter_ = gan.test(
            self.r, 
            self.dataset, 
            self.checkpoint, 
            batch_size=self.batch_size, 
            save_every=self.save_every, 
            return_batches=True, 
            return_hr=True,
        )

        for i, (LR, SR, HR) in enumerate(iter_):
            if self.denorm:
                LR = self.deprocess(LR)
                HR = self.deprocess(HR)
                SR = self.deprocess(SR)

            if self.groundtruth:
                SR = HR
            
            if calc_sh:
                t1 = time()
                C = SR.shape[-1]
                SR_sh = [np.stack([pysh.expand.SHExpandDH(img) for img in SR[..., c]], axis=0) for c in range(C)]
                SR_sh = np.stack(SR_sh, axis=-1)
                t2 = time()
                
                if self.measure_time:
                    print(f'sh-transform took {t2-t1:.2f}s')


            for name, metric in self.metrics.items():
                if i == 0:
                    metric.set_shape(HR.shape[1:])

                t1 = time()
                if metric.needs_sh:
                    metric.evaluate_SR_sh(i, LR, SR, SR_sh)
                else:
                    metric.evaluate_both(i, LR, SR, HR)
                    metric.evaluate_SR(i, LR, SR)
                t2 = time()

                if self.measure_time:
                    print(f'{name} took {t2-t1:.2f}s')

            print(i, flush=True)


        for metric in self.metrics.values():
            metric.finalize()


    def summarize(self, paths):
        pass


def main():
    Evaluation().run()


if __name__ == '__main__':
    main()