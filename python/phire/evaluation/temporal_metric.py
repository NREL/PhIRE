import numpy as np
from .base import EvaluationMethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class TemporalMetric(EvaluationMethod):

    def __init__(self, metric, label=None):
        super(TemporalMetric, self).__init__()
        self.metric = metric
        self.ts = []
        self.label = label
        self.no_groundtruth = True
    
    def evaluate_both(self, i, LR, SR, HR):
        self.ts.append(self.metric(SR,HR))


    def finalize(self):
        ts = np.concatenate(self.ts, axis=0)
        np.save(self.dir / 'losses.npy', ts, allow_pickle=False)


    def summarize(self, paths, outdir):
        data = {name: np.load(paths[name] / 'losses.npy') for name in paths}
        C = data[list(data.keys())[0]].shape[-1]

        with sns.plotting_context('paper'), sns.axes_style('whitegrid'), sns.color_palette('deep'):
            label = self.label if self.label else 'loss'

            for c in range(C):
                N = min([data[name].shape[0] for name in data])
                df = pd.DataFrame.from_dict({name: data[name][:N, c] for name in data})
                df = pd.melt(df, var_name='model', value_name=label)

                #g = sns.displot(x=label, hue='model', bins=35, height=3.0, aspect=1.0, data=df, palette=['C1', 'C2', 'C3', 'C4'])
                g = sns.displot(x=label, hue='model', bins=35, height=2.5, aspect=1.5, data=df)
                sns.move_legend(g, 'upper center', ncol=3, title=None)
                g.fig.savefig(outdir / f'error_histogram_{c}.png', bbox_inches='tight')
                g.fig.savefig(outdir / f'error_histogram_{c}.pdf', bbox_inches='tight')
                plt.close(g.fig)
                

            for name in data:
                for c in range(C):
                    y = data[name][..., c]
                    smoothing_window = 30*8
                    y_smoothed = np.convolve(y, np.ones(smoothing_window) / smoothing_window,  'valid')

                    fig, ax = plt.subplots(figsize=(4.2,2.5))
                    X = (smoothing_window//2 + np.arange(y_smoothed.shape[0])) / 8
                    ax.plot(X, y_smoothed, label='smoothed')
                    #ax.plot(X[2:-2], y[2:-2], alpha=0.5, ls=':', color='grey', label='raw')
                    ax.set_ylabel(label)
                    ax.set_xlabel('time [d]')
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    #ax.xaxis.set_major_formatter(plt.StrMethodFormatter('{x:.0f}'))
                    
                    fig.savefig(outdir / f'{name}_{c}.png', bbox_inches='tight')
                    fig.savefig(outdir / f'{name}_{c}.pdf', bbox_inches='tight')
                    
                    plt.close(fig)


