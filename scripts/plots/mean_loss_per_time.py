import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.ticker as ticker


DIR = '/data/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27'
layers = [196, 196, 196, 196]
losses = ['l2', 'l1', 'psnr', 'ssim']
labels = ['l2', 'l1', 'PSNR', 'SSIM']


def plot(layer, loss, label):
    losses = pd.read_csv(DIR + f'/layer{layer}_{loss}_loss.csv')
    scale = losses.alpha.values[0]

    # clip to 60h
    losses = losses[:20]

    fig, ax = plt.subplots(figsize=(4, 2.5))

    with sns.color_palette('deep'):
        x = np.arange(losses.shape[0]) + 1
        
        ax.fill_between(x, losses.metric_mean - losses.metric_std, losses.metric_mean + losses.metric_std, alpha=0.2)
        ax.fill_between(x, scale*(losses.content_loss_mean - losses.content_loss_std), scale*(losses.content_loss_mean + losses.content_loss_std), alpha=0.2)
        
        ax.plot(x, losses.metric_mean, label=label)
        ax.plot(x, scale*losses.content_loss_mean, label=f'ours')

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)*3}h'))
        ax.set_xlabel('time lag')
        ax.set_xticks(np.arange(losses.shape[0] + 1, step=4))
        ax.tick_params(which="both", bottom=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.set_ylabel('average loss')
        ax.legend()

    fig.savefig(DIR + f'/layer{layer}_{label}_loss_plot.pdf', bbox_inches='tight')


if __name__ == '__main__':
    sns.set_theme('paper')
    sns.set_style("whitegrid")

    for layer, loss, label in zip(layers, losses, labels):
        plot(layer, loss, label)