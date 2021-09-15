from .base import EvaluationMethod
import numpy as np
from glob import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Histogram(EvaluationMethod):

    def __init__(self, yx, patch_size):
        super(Histogram, self).__init__()
        self.values = {}
        self.yx = yx
        self.patch_size = patch_size

        if self.patch_size[0]*self.patch_size[1] > 25:
            print('[WARNING] patches larger than 5x5 might lead to excessive memory and disk usage (Histogram)')
    

    def evaluate_SR(self, idx, LR, SR):
        C = SR.shape[-1]
        for c in range(C):
            if c not in self.values:
                self.values[c] = []

            patch = SR[
                :, 
                self.yx[0] : self.yx[0] + self.patch_size[0], 
                self.yx[1] : self.yx[1] + self.patch_size[1], 
                c
            ]
            self.values[c].append(patch.flatten())  # guaranteed to copy

    
    def finalize(self):
        for c in self.values:
            data = np.concatenate(self.values[c])
            np.savetxt(self.dir / f'raw_channel{c}.csv', data)

    
    def summarize(self, paths, outdir):
        p = paths[next(iter(paths))]
        C = len(glob(str(p / 'raw_channel*.csv')))

        for c in range(C):
            data = {name: np.loadtxt(path / f'raw_channel{c}.csv') for name, path in paths.items()}
    
            quantiles = {}
            for name in data:
                percentiles = [2, 9, 25, 50, 75, 91, 98]
                q = np.percentile(data[name], percentiles)
                quantiles[name] = dict(zip(percentiles, q))

            with open(outdir / f'quantiles_channel{c}.json', 'w') as f:
                json.dump(quantiles, f, indent=4)

            data_df = pd.DataFrame.from_dict(data)
            data_df = pd.melt(data_df, var_name='model')

            with sns.plotting_context('paper'), sns.axes_style('whitegrid'), sns.color_palette('deep'):

                # box plot
                g = sns.catplot(x='value', y='model', kind='box', orient='h', height=5.5, 
                    showfliers=False, 
                    showmeans=True,
                    whis=(9,91),
                    data=data_df)
                g.set(xscale='symlog')
                
                g.fig.savefig(outdir / f'boxplot_channel{c}.png', bbox_inches='tight')
                g.fig.savefig(outdir / f'boxplot_channel{c}.pdf', bbox_inches='tight')
                plt.close(g.fig)

                # kde
                g = sns.displot(x='value', hue='model', kind='kde', data=data_df, bw_adjust=.6, clip=(-1e-4, 1.0e-4), cut=0) 
                g.set(yscale='log')
                g.fig.savefig(outdir / f'kde_channel{c}.png', bbox_inches='tight')
                g.fig.savefig(outdir / f'kde_channel{c}.pdf', bbox_inches='tight')
                plt.close(g.fig)

                means = {name: data[name].reshape((-1,) + self.patch_size).mean(axis=(1,2)) for name in data}

                fig, ax = plt.subplots(figsize=(8, 4.0))
                for name in means:
                    y = means[name][:200]
                    x = np.arange(y.shape[0])
                    ax.plot(x, y, label=name, alpha=0.7)

                ax.legend()
                ax.set_yscale('symlog')

                fig.savefig(outdir / f'timeseries_channel{c}.png', bbox_inches='tight')
                fig.savefig(outdir / f'timeseries_channel{c}.pdf', bbox_inches='tight')
                plt.close(fig)