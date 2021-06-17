import numpy as np
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pyshtools as pysh
from spharm import Spharmt

from .PhIREGANs import PhIREGANs

class PowerSpectrum:

    def __init__(self, H,W):
        self.spharm = Spharmt(H,W)
        self.spectrums_SR = []
        self.spectrums_HR = []

    
    def set_dir(self, dir):
        self.dir = dir


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
        plt.plot(np.arange(SR_spectrum.shape[0])**(-5/3) + HR_spectrum[0], '--')

        plt.legend()

        plt.savefig(self.dir / 'spectrum.png')


class Visualize:

    def __init__(self):
        self.n = 0


    def set_dir(self, dir):
        self.outdir = dir


    def evaluate(self, i, batch_LR, batch_SR, batch_HR):
        for img_LR, img_SR, img_HR in zip(batch_LR, batch_SR, batch_HR):

            directory = f'{self.outdir}/{self.n}'
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


class Evaluation:

    def __init__(self):
        self.dataset = sorted(glob('/data/stengel/HR/patches_train_1979_1990.0.tfrecords'))    
        self.checkpoint = '/data/results/models/mse-20210608-094931/training/gan-17'

        ##########################################

        self.mean_log1p, self.std_log1p = [0.008315503, 0.0028762482], [0.5266841, 0.5418187]
        self.mean, self.std = [2.0152406e-08, 2.1581373e-07], [2.8560082e-05, 5.0738556e-05]

        self.save_every = 100
        self.batch_size = 64
        self.r = [2,2]

        self.metrics = {
            'power-spectrum': PowerSpectrum(96, 96),
            'visualize': Visualize()
        }

    
    def deprocess(self, batch):
        y = batch * self.std_log1p + self.mean_log1p
        y =  np.sign(y) * np.expm1(np.fabs(y))
        return y*self.std + self.mean


    def run(self):
        gan = PhIREGANs('eval', mu_sig=[[0,0], [1,1]], print_every=40, compression='ZLIB')
        main_dir = Path('ABC')

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
                metric.evaluate(i, batch_LR, batch_SR, batch_HR)

        for metric in self.metrics.values():
            metric.finalize()


def main():
    Evaluation().run()


if __name__ == '__main__':
    main()