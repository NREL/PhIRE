import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sns.set_theme('paper')

    data = pd.read_csv('histogram.csv')
    tmp = data.values[:, 1:].reshape(-1, 4, 2).sum(axis=1)
    vort, div = tmp[:, 0], tmp[:, 1]

    plt.figure(figsize=(6,3.0))

    with sns.color_palette('deep'):
        plt.bar(data['Unnamed: 0'].values[::4] + 8e-5, height=vort, width=8e-5, label='vorticity')
        plt.bar(data['Unnamed: 0'].values[::4] + 8e-5, height=div, width=8e-5, label='divergence')

    plt.ylabel('count')
    plt.yscale('log')
    plt.xlabel('value')
    plt.legend()
    plt.savefig('histogram.pdf', bbox_inches='tight')