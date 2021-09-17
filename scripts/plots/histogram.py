import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == '__main__':
    sns.set_theme('paper')
    sns.set_style("whitegrid")

    data = pd.read_csv('histogram.csv')
    tmp = data.values[:, 1:].reshape(-1, 4, 2).sum(axis=1)
    vort, div = tmp[:, 0], tmp[:, 1]

    fig, ax = plt.subplots(figsize=(5.0,2.5))

    with sns.color_palette('deep'):
        ax.bar(data['Unnamed: 0'].values[::4] + 8e-5, height=vort, width=8e-5, label='vorticity')
        ax.bar(data['Unnamed: 0'].values[::4] + 8e-5, height=div, width=8e-5, label='divergence')

    ax.legend()

    ax.set_ylabel('count')
    ax.set_xlabel('value')
    ax.set_yscale('log')

    formatter = ticker.ScalarFormatter()
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    fig.savefig('histogram.pdf', bbox_inches='tight')