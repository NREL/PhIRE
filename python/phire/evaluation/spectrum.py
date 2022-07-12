from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from spharm import Spharmt
import pyshtools as pysh
import seaborn as sns

from .base import EvaluationMethod


def _plot_spectrum_200(data):
    #fig, ax = plt.subplots(figsize=(5.5, 2.75))
    fig, ax = plt.subplots(figsize=(4,2.5))


    for name, spectrum in data.items():
        x = np.arange(1, 1+spectrum.shape[0])

        # only plot wavenumber 200+
        spectrum = spectrum[200:]
        x = x[200:]

        ax.plot(x, spectrum, label=name)

    ax.loglog()

    ax.legend()
    ax.set_xlabel('wavenumber')
    ax.set_ylabel('energy')

    ax.set_xticks([200, 300, 400, 500, 600])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    return fig, ax


def _plot_spectrum_high(data):
    fig, ax = plt.subplots(figsize=(5.5, 2.75))

    for name, spectrum in data.items():
        x = np.arange(1, 1+spectrum.shape[0])

        # only plot wavenumber 550+
        spectrum = spectrum[550:]
        x = x[550:]

        ax.plot(x, spectrum, label=name)

    ax.loglog()

    ax.legend()
    ax.set_xlabel('wavenumber')
    ax.set_ylabel('energy')

    #ax.set_xticks([200, 300, 400, 500, 600])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    return fig, ax


def _plot_spectrum_low(data):
    fig, ax = plt.subplots(figsize=(5.5, 2.75))

    for name, spectrum in data.items():
        x = np.arange(1, 1+spectrum.shape[0])

        spectrum = spectrum[10:200]
        x = x[10:200]

        ax.plot(x, spectrum, label=name)

    ax.set_yscale('log')

    ax.legend()
    ax.set_xlabel('wavenumber')
    ax.set_ylabel('energy')

    #ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, ])
    #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    return fig, ax


class PowerSpectrum(EvaluationMethod):

    def __init__(self, H,W):
        super(PowerSpectrum, self).__init__()
        #self.spharm = Spharmt(W,H, legfunc='stored')
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


    def summarize(self, paths, outdir):
        p = paths[next(iter(paths))]
        C = len(glob(str(p / 'spectrum_channel_*.csv')))
        
        with sns.plotting_context('paper'), sns.axes_style('whitegrid'), sns.color_palette('deep'):
        
            # energy
            energy_spectrums  = {name: np.loadtxt(path / 'energy_spectrum.csv') for name, path in paths.items()}
            
            fig, ax = _plot_spectrum_200(energy_spectrums)
            fig.savefig(outdir / 'energy_spectrum.pdf', bbox_inches='tight')
            fig.savefig(outdir / 'energy_spectrum.png', bbox_inches='tight')
            plt.close(fig)

            fig, ax = _plot_spectrum_low(energy_spectrums)
            fig.savefig(outdir / 'energy_spectrum_lowend.pdf', bbox_inches='tight')
            fig.savefig(outdir / 'energy_spectrum_lowend.png', bbox_inches='tight')
            plt.close(fig)

            fig, ax = _plot_spectrum_high(energy_spectrums)
            fig.savefig(outdir / 'energy_spectrum_highend.pdf', bbox_inches='tight')
            fig.savefig(outdir / 'energy_spectrum_highend.png', bbox_inches='tight')
            plt.close(fig)

            # individual channels
            for c in range(C):
                spectrums  = {name: np.loadtxt(path / f'spectrum_channel_{c}.csv') for name, path in paths.items()}
                
                fig, ax = _plot_spectrum_200(spectrums)
                fig.savefig(outdir / f'spectrum_channel_{c}.pdf', bbox_inches='tight')
                fig.savefig(outdir / f'spectrum_channel_{c}.png', bbox_inches='tight')
                plt.close(fig)

                fig, ax = _plot_spectrum_low(spectrums)
                fig.savefig(outdir / f'spectrum_channel_{c}_lowend.pdf', bbox_inches='tight')
                fig.savefig(outdir / f'spectrum_channel_{c}_lowend.png', bbox_inches='tight')
                plt.close(fig)

                fig, ax = _plot_spectrum_high(spectrums)
                fig.savefig(outdir / f'spectrum_channel_{c}_highend.pdf', bbox_inches='tight')
                fig.savefig(outdir / f'spectrum_channel_{c}_highend.png', bbox_inches='tight')
                plt.close(fig)

            