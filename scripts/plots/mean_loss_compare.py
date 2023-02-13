import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.ticker as ticker


DIR1 = '/data/final_rp_models/autoencoder_2022-07-19_1049/epoch39'
DIR2 = '/data/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27'

layer = 196
loss = 'l2'

def plot():
    losses1 = pd.read_csv(DIR1 + f'/layer{layer}_{loss}_loss.csv')
    scale1 = losses1.alpha.values[0]

    losses2 = pd.read_csv(DIR2 + f'/layer{layer}_{loss}_loss.csv')
    scale2 = losses2.alpha.values[0]

    # clip to 60h
    losses1 = losses1[:20]
    losses2 = losses2[:20]


    fig, ax = plt.subplots(figsize=(4, 2.5))

    with sns.color_palette('deep'):
        x = np.arange(losses1.shape[0]) + 1
        
        ax.fill_between(x, scale1*(losses1.content_loss_mean - losses1.content_loss_std), scale1*(losses1.content_loss_mean + losses1.content_loss_std), alpha=0.2)
        ax.fill_between(x, scale2*(losses2.content_loss_mean - losses2.content_loss_std), scale2*(losses2.content_loss_mean + losses2.content_loss_std), alpha=0.2)
        
        ax.plot(x, scale1*losses1.content_loss_mean, label=f'autoencoder')
        ax.plot(x, scale2*losses2.content_loss_mean, label=f'atmodist')


        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)*3}h'))
        ax.set_xlabel('time lag')
        ax.set_xticks(np.arange(losses1.shape[0] + 1, step=4))
        ax.tick_params(which="both", bottom=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.set_ylabel('average loss')
        ax.legend()

    fig.savefig(DIR1 + f'/autoencoder_vs_atmodist.pdf', bbox_inches='tight')


if __name__ == '__main__':
    sns.set_theme('paper')
    sns.set_style("whitegrid")

    plot()