import xarray as xr
import numpy as np
from scipy.fft import rfft, irfft
from phire.utils import lanczos_filter_xr


def aggregate_dataset(filename):
    ds = xr.open_dataset(filename, engine='cfgrib')
    daily_means = ds.u.coarsen(time=24).mean()
    averaged = daily_means.mean(dim=['isobaricInhPa', 'longitude'])
    return averaged


def seasonal_variation(x):
    x = x.dropna(dim='time')
    mean_per_day = x.groupby('time.dayofyear').mean()

    freq = rfft(mean_per_day.values)
    freq[3:] = 0
    
    lowpassed = irfft(freq, 366)
    return lowpassed


def main():
    datasets = [aggregate_dataset(f'/data/ERA5/jetstream/u_wind_{y}_m{m}.grib') for y in range(1979, 2020) for m in range(1,13)]
    data = xr.concat(datasets, dim='time')

    filtered = lanczos_filter_xr(data, 61, 10, 'time', center=True)
    latitudes = filtered.idxmax(dim='latitude')
    speeds = filtered.loc[:, latitudes.fillna(75.)]

    latitude_seasonal = seasonal_variation(latitudes)
    speed_seasonal = seasonal_variation(speeds)

    latitude_anomaly = latitudes - latitude_seasonal[latitudes.time.dt.dayofyear - 1]
    speed_anomaly = speeds - speed_seasonal[speeds.time.dt.dayofyear - 1]

    np.savetxt('latitude_anomaly.csv', latitude_anomaly.values)
    np.savetxt('speed_anomaly.csv', speed_anomaly.values)
    np.savetxt('time.csv', latitude_anomaly.time.values)
    np.savetxt('latitudes.csv', latitudes.values)
    np.savetxt('speeds.csv', speeds.values)
    np.savetxt('raw.csv', filtered.values)
    np.savetxt('latitude_seasonal.csv', latitude_seasonal)
    np.savetxt('speed_seasonal.csv', speed_seasonal)


if __name__ == '__main__':
    main()