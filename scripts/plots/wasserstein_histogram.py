import matplotlib
matplotlib.use('agg')
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import json
import pandas as pd
import numpy as np
import json
from pathlib import Path
import pkg_resources

def main():
    

    for c in range(2):
        files = glob(f'./*/wasserstein_dist_channel{c}.json')
        data = []
        for file in files:
            with open(file, 'r') as f:
                dct = json.load(f)
            
            dct['city'] = Path(file).parts[-2].replace('_', ' ').title()
            data.append(dct)
            
        
        df_wide = pd.DataFrame(data)

        df = df_wide.melt(value_name='wasserstein distance', var_name='model', id_vars=['city'])
        df = df[df.model != 'ground truth']
        
        with sns.plotting_context('paper'), sns.axes_style('whitegrid'), sns.color_palette('deep'):
            g = sns.displot(df, x='wasserstein distance', hue='model', kind='hist', bins=30, log_scale=True, height=3, aspect=1.5)        
            g.fig.savefig(f'wasserstein_hist_channel{c}.png', bbox_inches='tight')
            g.fig.savefig(f'wasserstein_hist_channel{c}.pdf', bbox_inches='tight')


        worser = int(np.sum(df_wide.ours > 1.1*df_wide.mse))
        better = int(np.sum(df_wide.mse > 1.1*df_wide.ours))
        equal = len(df_wide) - worser - better
        with open(f'stats_channel{c}.json', 'w') as f:
            json.dump({'worser': worser, 'equal': equal, 'better': better}, f, indent=4)


        cities = pd.read_csv(pkg_resources.resource_filename('phire', 'data/cities.csv'))
        
        def plot(col):
            merged = df_wide.sort_values(col).merge(cities, left_on='city', right_on='city_ascii')
            overperformer = merged.head(30)
            underperformer = merged.tail(30)

            fig, ax = plt.subplots(figsize=(10,5), subplot_kw={'projection':ccrs.Robinson()})
            #ax.coastlines()
            ax.stock_img()

            for _, city in overperformer.iterrows():
                ax.plot(city.lng, city.lat, 'g*', transform=ccrs.Geodetic())

            for _, city in underperformer.iterrows():
                ax.plot(city.lng, city.lat, 'r*', transform=ccrs.Geodetic())
            
            fig.savefig(f'outliers_{col}_channel{c}.png', bbox_inches='tight')
            fig.savefig(f'outliers_{col}_channel{c}.pdf', bbox_inches='tight')

        plot('ours')
        plot('mse')

if __name__ == '__main__':
    main()