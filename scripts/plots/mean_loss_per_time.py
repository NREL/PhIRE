import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.ticker as ticker


DIR = '/data/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27'
layer = 196

if __name__ == '__main__':
    sns.set_theme('paper')

    losses = pd.read_csv(DIR + f'/layer{layer}_loss.csv')
    scale = float(Path(DIR + f'/layer{layer}_scale.txt').read_text())

    fig, ax = plt.subplots(figsize=(6, 3.0))

    with sns.color_palette('deep'):
        x = np.arange(losses.shape[0]) + 1
        
        ax.fill_between(x, losses.mse_mean - losses.mse_std, losses.mse_mean + losses.mse_std, alpha=0.2)
        ax.fill_between(x, scale*(losses.layer_mean - losses.layer_std), scale*(losses.layer_mean + losses.layer_std), alpha=0.2)
        
        ax.plot(x, losses.mse_mean, label='mse')
        ax.plot(x, scale*losses.layer_mean, label=f'ours')

        #fmt = dict(fmt=':', capsize=3, capthick=1)
        #ax.errorbar(x, losses.mse_mean, losses.mse_std / 2, label='mse', **fmt)
        #ax.errorbar(x, scale*losses.layer_mean, scale*losses.layer_std / 2, label=f'TempoDist (layer-{layer})', **fmt)
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)*3}h'))
        ax.set_xlabel('time difference between samples')
        ax.set_xticks(np.arange(losses.shape[0] + 1, step=4))
        ax.tick_params(which="both", bottom=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.set_ylabel('average loss')
        ax.legend()

    fig.savefig(DIR + f'/layer{layer}_loss_plot.pdf', bbox_inches='tight')