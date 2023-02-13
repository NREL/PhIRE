import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import PercentFormatter


DIR = '/data/final_rp_models/rnet-small-abla-31c_2021-10-10_1702/'

if __name__ == '__main__':
    sns.set_theme('paper')
    sns.set_style("whitegrid")

    train = pd.read_csv(DIR + '/training.csv', sep=' ')
    eval = pd.read_csv(DIR + '/evaluation.csv')

    train = train.reset_index()
    train['epoch'] = train['index']

    
    with sns.color_palette('deep'):
        # loss plot
        fig0, ax0 = plt.subplots(figsize=(3, 2.0))
        ax0.plot(train.epoch - 19, train.loss, label='training set')
        ax0.plot(eval.epoch + 1, eval.loss, label='evaluation set')
        
        ax0.axvspan(-19, 1, alpha=0.2, facecolor='lightgrey', edgecolor='grey', hatch='/')
        trans = blended_transform_factory(ax0.transData, ax0.transAxes)
        ax0.text(-9, 0.5, 'pretraining', horizontalalignment='center', transform=trans)

        ax0.legend()
        ax0.set_xlabel('epoch')
        ax0.set_ylabel('loss')


        # acc plot
        fig1, ax1 = plt.subplots(figsize=(3.0, 2.0))
        ax1.plot(train.epoch - 19, train.categorical_accuracy, label='training set')
        ax1.plot(eval.epoch + 1, eval.accuracy, label='evaluation set')
        
        ax1.axvspan(-19, 1, alpha=0.2, facecolor='lightgrey', edgecolor='grey', hatch='/')
        trans = blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.text(-9, 0.5, 'pretraining', horizontalalignment='center', transform=trans)

        ax1.legend()
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
        
    fig0.savefig(DIR + '/rplearn_loss.pdf', bbox_inches='tight')
    fig1.savefig(DIR + '/rplearn_accuracy.pdf', bbox_inches='tight')