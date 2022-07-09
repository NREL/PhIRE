import numpy as np
import matplotlib.pyplot as plt
import sys


def plot(cm, mdir):
    cm_normalized = cm / np.sum(cm, axis=0, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    im = ax.imshow(cm_normalized, interpolation='nearest', aspect='equal', vmin=0, vmax=1)
    fig.colorbar(im, fraction=0.046, pad=0.05)
    ax.set_xlabel('predicted time lag')
    ax.set_ylabel('actual time lag')
    ax.xaxis.set_major_formatter(lambda x, pos: f'{3 + x*3:.0f}h')
    ax.yaxis.set_major_formatter(lambda x, pos: f'{3 + x*3:.0f}h')

    fig.savefig(mdir + '/confusion_matrix.png', bbox_inches='tight')
    fig.savefig(mdir + '/confusion_matrix.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    mdir = sys.argv[1]
    print(mdir)
    cm = np.loadtxt(mdir + '/confusion_matrix.csv')
    plot(cm, mdir)


if __name__ == '__main__':
    main()