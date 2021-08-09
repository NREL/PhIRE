from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from spharm import Spharmt
import pyshtools as pysh


from .base import EvaluationMethod


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
        p = paths[next(iter(paths))]
        C = len(glob(str(p / 'spectrum_channel_*.csv')))

        for c in range(C):
            plt.figure(figsize=(10,5))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('wavenumber')
            plt.ylabel('energy')

            for name, path in paths.items():
                spectrum = np.loadtxt(path / f'spectrum_channel_{c}.csv')
                x = np.arange(1, 1+spectrum.shape[0])
                plt.plot(x, spectrum, label=name)

            plt.plot(x, x**(-5/3) * spectrum[0], '--')
            plt.legend()

            plt.savefig(outdir / f'spectrum_channel_{c}.png')

        plt.figure(figsize=(10,5))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('wavenumber')
        plt.ylabel('energy')

        for name, path in paths.items():
            spectrum = np.loadtxt(path / 'energy_spectrum.csv')
            x = np.arange(1, 1+spectrum.shape[0])
            plt.plot(x, spectrum, label=name)

        plt.plot(x, x**(-5/3) * spectrum[0], '--')
        plt.legend()

        plt.savefig(outdir / 'energy_spectrum.png')