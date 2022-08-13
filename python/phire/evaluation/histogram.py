from .base import EvaluationMethod
import numpy as np
from glob import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import wasserstein_distance


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
            np.save(self.dir / f'raw_channel{c}.npy', data)

    
    def summarize(self, paths, outdir):
        p = paths[next(iter(paths))]
        C = len(glob(str(p / 'raw_channel*.npy')))

        for c in range(C):
            data = {name: np.load(path / f'raw_channel{c}.npy') for name, path in paths.items()}
            N = min(data[k].shape[0] for k in data)
            data = {k: data[k][:N] for k in data}
            channel_name = ['divergence [s\u207B\u00B9]', 'vorticity [s\u207B\u00B9]'][c] 

            # quantiles
            quantiles = {}
            for name in data:
                percentiles = [2, 9, 25, 50, 75, 91, 98]
                q = np.percentile(data[name], percentiles)
                quantiles[name] = dict(zip(percentiles, q))

            with open(outdir / f'quantiles_channel{c}.json', 'w') as f:
                json.dump(quantiles, f, indent=4)

            data_df = pd.DataFrame.from_dict(data)
            data_df = pd.melt(data_df, var_name='model', value_name=channel_name)

            # Wasserstein distance
            if 'ground truth' in data:
                wdist =  {name: wasserstein_distance(data[name], data['ground truth']) for name in data}
                with open(outdir / f'wasserstein_dist_channel{c}.json', 'w') as f:
                    json.dump(wdist, f, indent=4)

            with sns.plotting_context('paper'), sns.axes_style('whitegrid'), sns.color_palette('deep'):

                # box plot
                g = sns.catplot(x=channel_name, y='model', kind='box', orient='h',  height=1.5, aspect=3.25,
                    showfliers=False, 
                    showmeans=True,
                    meanprops={'markerfacecolor': 'dimgray', 'markeredgecolor': 'dimgray'},
                    whis=(9,91),
                    data=data_df)
                g.set(xscale='symlog')
                g.ax.xaxis.set_major_locator(plt.MaxNLocator(9, steps=[1,2,4], min_n_ticks=4))#plt.MultipleLocator(1e-5))
                formatter = ticker.ScalarFormatter()
                formatter.set_powerlimits((-1, 1))
                g.ax.xaxis.set_major_formatter(formatter)
                
                g.fig.savefig(outdir / f'boxplot_channel{c}.png', bbox_inches='tight')
                g.fig.savefig(outdir / f'boxplot_channel{c}.pdf', bbox_inches='tight')
                plt.close(g.fig)

                # kde
                g = sns.displot(x=channel_name, hue='model', kind='kde', data=data_df, bw_adjust=.5, clip=(-7e-5, 7e-5), cut=0.5, height=2.5, aspect=1.25) 
                sns.move_legend(g, 'upper right', title=None)
                g.set(yscale='log')
                for line2d in g.ax.get_lines()[:-1]:
                    line2d.set_linestyle(':')
                    line2d.set_linewidth(1.3)
                    line2d.set_alpha(0.9)
                g.ax.get_lines()[-1].set_linewidth(2.0)
                
                formatter = ticker.ScalarFormatter()
                formatter.set_powerlimits((-1, 1))
                g.ax.xaxis.set_major_formatter(formatter)
                
                g.fig.savefig(outdir / f'kde_channel{c}.png', bbox_inches='tight')
                g.fig.savefig(outdir / f'kde_channel{c}.pdf', bbox_inches='tight')
                plt.close(g.fig)

                """
                # timeseries
                timeseries = {name: np.fabs(data[name] - data['groundtruth'])[:100] for name in data}
                fig, ax = plt.subplots(figsize=(8, 4.0))
                for name in timeseries:
                    y = timeseries[name]
                    x = np.arange(y.shape[0])
                    ax.plot(x, y, label=name, alpha=0.6)
                    ax.fill_between(x, 0, y, alpha=0.2)


                ax.legend()
                ax.set_yscale('symlog')

                fig.savefig(outdir / f'timeseries_channel{c}.png', bbox_inches='tight')
                fig.savefig(outdir / f'timeseries_channel{c}.pdf', bbox_inches='tight')
                plt.close(fig)
                """