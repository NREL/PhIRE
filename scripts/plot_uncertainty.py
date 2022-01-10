import matplotlib
matplotlib.use('Agg')  # this results in considerable plotting speedup and enables multiprocessing + non-interactive work


import numpy as np
import scipy
import os
import tarfile
from absl import flags, app

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


FLAGS = flags.FLAGS
flags.DEFINE_bool('interpolate', False, 'interpolate images')



def plot_scalarfield(field, name, extent=None, vmin=None, vmax=None, cmap=None):
    extent = (0 + 11.25, 360 + 11.25, 90 - 11.25, -90 + 11.25) if extent is None else extent
    

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent((-180, 179.99, 65, -65), ccrs.PlateCarree())
    ax.coastlines()

    # interpolate
    if FLAGS.interpolate:
        x = np.linspace(0,1 + 1/field.shape[1], field.shape[1] + 1)
        y = np.linspace(0,1,field.shape[0])
        padded = np.pad(field, ((0,0), (0,1)), 'wrap')
        field_int = scipy.interpolate.interp2d(x,y, padded, 'cubic')
        upscaled = field_int(np.linspace(0,1,field.shape[1]*5), np.linspace(0,1,field.shape[0]*5))
    else:
        upscaled = field

    # show field
    mappable = ax.imshow(upscaled, transform=ccrs.PlateCarree(), extent=extent, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(mappable, orientation='horizontal', pad=0.05, aspect=40)
    
    lat = [-60, -60, -60+22.5, -60+22.5, -60]
    lon = [-175, -175+22.5, -175+22.5, -175, -175]
    ax.plot(lon, lat, transform=ccrs.PlateCarree(), color='red')

    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def plot_preds(preds, dir, delta_T):
    pred_labels = np.argmax(preds, axis=-1)
    entropy = -np.sum(preds * np.log(preds), axis=-1)

    plot_scalarfield(np.mean(preds[..., (delta_T-1)], axis=0), dir + '/proba.png', vmin=0., vmax=1., cmap='PiYG')
    plot_scalarfield(np.mean(preds[..., (delta_T-2)] + preds[..., (delta_T-1)] + preds[..., delta_T], axis=0), dir + '/top3_proba.png', vmin=0., vmax=1., cmap='PiYG')
    
    plot_scalarfield(np.mean(pred_labels == (delta_T-1), axis=0), dir + '/accuracy.png', vmin=0., vmax=1., cmap='PiYG')
    plot_scalarfield(np.mean((pred_labels >= (delta_T-2)) & (pred_labels <= delta_T), axis=0), dir + '/top3_accuracy.png', vmin=0., vmax=1., cmap='PiYG')
   
    plot_scalarfield(np.mean(entropy, axis=0), dir + '/entropy.png')
    

def main(argv):
    delta_T = 12
    outdir = f'uncertainty/dt{delta_T}'

    preds = np.load(f'uncertainty/raw_{delta_T}.npy')

    os.makedirs(outdir + '/all', exist_ok=True)
    plot_preds(preds, outdir + '/all', delta_T)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dez']
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Months
    i = 0
    for M, n_days in zip(months, days):
        os.makedirs(outdir + '/' + M, exist_ok=True)
        plot_preds(preds[i:i+n_days*8], outdir + '/' + M, delta_T)
        i += n_days*8

    # Time
    for i, label in enumerate(['00', '03', '06', '09', '12', '15', '18', '21']):
        os.makedirs(outdir + f'/by_time/{label}', exist_ok=True)
        plot_preds(preds[i::8], outdir + f'/by_time/{label}', delta_T)


    # Warm / Cold Season
    preds = np.pad(preds, ((0, 90), (0,0),(0,0),(0,0)), 'wrap')
    warm_season = preds[90*8:(90+183)*8]
    cold_season = preds[(90+183)*8:]
    os.makedirs(outdir + '/warm', exist_ok=True)
    os.makedirs(outdir + '/cold', exist_ok=True)
    plot_preds(warm_season, outdir + '/warm', delta_T)
    plot_preds(cold_season, outdir + '/cold', delta_T)

    # Bundle everything up
    with tarfile.open(outdir + '.tar.gz', "w:gz") as tar:
        tar.add(outdir, arcname=os.path.basename(outdir))


if __name__ == '__main__':
    app.run(main)