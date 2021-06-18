import numpy as np
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pyshtools as pysh
from spharm import Spharmt

from .PhIREGANs import PhIREGANs



class EvaluationMethod:

    def set_dir(self, dir):
        self.dir = dir


    def set_shape(self, shape):
        self.shape = shape


    def finalize(self):
        pass


class PowerSpectrum(EvaluationMethod):

    def __init__(self, H,W):
        self.spharm = Spharmt(H,W)
        self.spectrums_SR = []
        self.spectrums_HR = []


    def _calc_spectrum(self, div_vort):
        sh = self.spharm.grdtospec(div_vort)
        u,v = self.spharm.getuv(sh[..., 1], sh[..., 0])
        energy = 0.5 * (u**2 + v**2)
        
        energy_sh = pysh.expand.SHExpandDH(energy)
        spectrum = pysh.spectralanalysis.spectrum(energy_sh)
        return spectrum


    def evaluate(self, i, batch_LR, batch_SR, batch_HR):
        for img in batch_SR:
            spectrum = self._calc_spectrum(img)
            self.spectrums_SR.append(spectrum)

        for img in batch_HR:
            spectrum = self._calc_spectrum(img)
            self.spectrums_HR.append(spectrum)


    def finalize(self):
        SR_spectrum = np.mean(np.stack(self.spectrums_SR), axis=0)
        HR_spectrum = np.mean(np.stack(self.spectrums_HR), axis=0)

        np.savetxt(self.dir / 'SR_spectrum.csv', SR_spectrum)
        np.savetxt(self.dir / 'HR_spectrum.csv', HR_spectrum)

        plt.figure(figsize=(7,4))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('wavenumber')
        plt.ylabel('energy')

        plt.plot(SR_spectrum, label='SR')
        plt.plot(HR_spectrum, label='groundtruth')
        plt.plot(np.arange(1, 1+SR_spectrum.shape[0])**(-5/3) * HR_spectrum[0], '--')

        plt.legend()

        plt.savefig(self.dir / 'spectrum.png')


class Visualize(EvaluationMethod):

    def __init__(self):
        self.n = 0


    def evaluate(self, i, batch_LR, batch_SR, batch_HR):
        for img_LR, img_SR, img_HR in zip(batch_LR, batch_SR, batch_HR):

            directory = f'{self.dir}/{self.n}'
            os.makedirs(directory)

            plt.imsave(f'{directory}/div.png', np.hstack([img_SR[...,0], img_HR[...,0]]))
            plt.imsave(f'{directory}/vort.png', np.hstack([img_SR[...,1], img_HR[...,1]]))
            
            plt.imsave(f'{directory}/SR_div.png', img_SR[...,0])
            plt.imsave(f'{directory}/SR_vort.png', img_SR[...,1])

            plt.imsave(f'{directory}/HR_div.png', img_HR[...,0])
            plt.imsave(f'{directory}/HR_vort.png', img_HR[...,1])

            plt.imsave(f'{directory}/LR_div.png', img_LR[...,0])
            plt.imsave(f'{directory}/LR_vort.png', img_LR[...,1])

            self.n += 1


class Semivariogram(EvaluationMethod):

    def __init__(self, lat_min = 0, lat_max = 180, long_min = 0, long_max=360, direction=None, n_samples=100):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.long_min = long_min
        self.long_max = long_max

        self.direction = direction
        self.n_samples = n_samples
        self.lag_range = range(30, 300, 20)


    def set_shape(self, shape):
        self.C = shape[-1]
        self.SR_diffs = [[[] for _ in self.lag_range] for _ in range(self.C)]
        self.HR_diffs = [[[] for _ in self.lag_range] for _ in range(self.C)]


    def _sample(self, img, r_degree):
        sh = pysh.expand.SHExpandDH(img)

        if self.direction:
            lat_dir = self.direction[0]
            lat = np.random.uniform(0 + min(0, r_degree*lat_dir), 180 - max(0, r_degree*lat_dir), size=self.n_samples)
            lon = np.random.uniform(0, 360, size=self.n_samples)
            directions = np.tile(self.directions, (self.n_samples, 1))
        else:
            lat = np.random.uniform(0, 180, size=self.n_samples)
            lon = np.random.uniform(0, 360, size=self.n_samples) 
            directions = np.random.normal(size=2*self.n_samples).reshape((self.n_samples, 2))

        directions = r_degree * (directions / np.linalg.norm(directions, axis=1)[:, None])

        y1 = pysh.expand.MakeGridPoint(sh, lat=lat, lon=lon)
        
        lat2 = lat+directions[:,0]
        lat2[(lat2 < 0) | (lat2 > 180)] = (lat - directions[:,0])[(lat2 < 0) | (lat2 > 180)]
        y2 = pysh.expand.MakeGridPoint(sh, lat=lat2, lon=lon+directions[:,1])

        return y1, y2


    def _calc(self, batch, storage):
        for img in batch:
            for c in range(self.C):
                for i, radius in enumerate(self.lag_range):
                    y1, y2 = self._sample(img[..., 0], radius * (360 / 40075))
                    sq_diffs = (y1 - y2)**2
                    storage[c][i] += list(sq_diffs)


    def evaluate(self, i, batch_LR, batch_SR, batch_HR):
        self._calc(batch_SR, self.SR_diffs)
        self._calc(batch_HR, self.HR_diffs)        


    def finalize(self):
        
        SR_diffs = np.asarray(self.SR_diffs)  # C x lag x N
        SR_means = np.mean(SR_diffs, axis=2)
        SR_stds = np.std(SR_diffs, axis=2)

        for c in range(self.C):
            np.savetxt(self.dir / f'SR_channel_{c}.csv', SR_means[c])

        HR_diffs = np.asarray(self.HR_diffs)  # C x lag x N
        HR_means = np.mean(HR_diffs, axis=2)
        HR_stds = np.std(HR_diffs, axis=2)

        for c in range(self.C):
            np.savetxt(self.dir / f'HR_channel_{c}.csv', HR_means[c])

        for c in range(self.C):
            plt.figure(figsize=(7,3))
            plt.yscale('log')
            plt.xscale('log')

            plt.plot([r for r in self.lag_range], SR_means[c],  label='SR')
            plt.plot([r for r in self.lag_range], HR_means[c], label='groundtruth')
            
            plt.xlabel('lag distance')
            plt.legend()

            plt.savefig(self.dir / f'channel_{c}.png')



class Evaluation:

    def __init__(self):
        self.dir = 'srgan/mse/gan17'
        self.dataset = sorted(glob('/data/stengel/HR/patches_train_1979_1990.0.tfrecords'))    
        self.checkpoint = '/data/results/models/mse-20210608-094931/training/gan-17'

        ##########################################

        self.mean_log1p, self.std_log1p = [0.008315503, 0.0028762482], [0.5266841, 0.5418187]
        self.mean, self.std = [2.0152406e-08, 2.1581373e-07], [2.8560082e-05, 5.0738556e-05]

        self.save_every = 500
        self.batch_size = 64
        self.r = [2,2]

        self.metrics = {
            'power-spectrum': PowerSpectrum(96, 96),
            'visualize': Visualize(),
            'semivariogram': Semivariogram()
        }

    
    def deprocess(self, batch):
        y = batch * self.std_log1p + self.mean_log1p
        y =  np.sign(y) * np.expm1(np.fabs(y))
        return y*self.std + self.mean


    def run(self):
        gan = PhIREGANs('eval', mu_sig=[[0,0], [1,1]], print_every=50, compression='ZLIB')
        main_dir = Path(self.dir)

        iter_ = gan.test(
            self.r, 
            self.dataset, 
            self.checkpoint, 
            batch_size=self.batch_size, 
            save_every=self.save_every, 
            return_batches=True, 
            return_hr=True,
        )

        for name, metric in self.metrics.items():
            os.makedirs(main_dir / name)
            metric.set_dir(main_dir / name)


        for i, (batch_LR, batch_SR, batch_HR) in enumerate(iter_):
            batch_LR = self.deprocess(batch_LR)
            batch_SR = self.deprocess(batch_SR)
            batch_HR = self.deprocess(batch_HR)

            for name, metric in self.metrics.items():
                if i == 0:
                    metric.set_shape(batch_HR.shape[1:])

                metric.evaluate(i, batch_LR, batch_SR, batch_HR)

        for metric in self.metrics.values():
            metric.finalize()


def main():
    Evaluation().run()


if __name__ == '__main__':
    main()