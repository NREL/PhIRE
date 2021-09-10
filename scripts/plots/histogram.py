import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sns.set_theme('paper')

    data = pd.read_csv('histogram.csv')
    tmp = data.values[:, 1:].reshape(-1, 4, 2).sum(axis=1)
    vort, div = tmp[:, 0], tmp[:, 1]

    plt.figure(figsize=(7,3.5))

    with sns.color_palette('deep'):
        plt.bar(data['Unnamed: 0'].values[::4] + 8e-5, height=vort, width=8e-5, label='Vorticity')
        plt.bar(data['Unnamed: 0'].values[::4] + 8e-5, height=div, width=8e-5, label='Divergence')

    plt.ylabel('Count')
    plt.yscale('log')
    plt.xlabel('Value')
    plt.legend()
    plt.savefig('histogram.pdf')