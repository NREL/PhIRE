import math
import calendar
import pvlib
import pdb
import h5py
import numpy as np
import pandas as pd
from math import pi
from multiprocessing import Pool, RawArray, Array
from itertools import product
import logging
import sys

#NEED TO ADD IN FOR VARIABLE LAT AND LON?

noon = 'clock'
var_dict = {}


def init_worker(input_array_ghi, input_array_sza,# input_array_pres,
                input_shape_ghi, input_shape_sza,# input_shape_pres,
                time):


    var_dict['input_array_ghi'] = input_array_ghi
    var_dict['input_array_sza'] = input_array_sza

    var_dict['input_shape_ghi'] = input_shape_ghi
    var_dict['input_shape_sza'] = input_shape_sza

    var_dict['time'] = time

    assert var_dict['input_array_ghi'].shape == var_dict['input_array_sza'].shape# == var_dict['input_array_pres'].shape

def worker_func(ind):
    # split index
    i, j, y = ind

    _lon = lon.loc[lon.id == j + 1].lon.values[0]
    #_lat = lat.loc[lat.id == i + 1].lat.values[0]

    year_start = y * _time_dim
    year_end = year_start + _time_dim
    if i == 0 and j == 0:
        logging.debug(f'[{y}][{i}][{j}] year_start: {year_start}, year_end: {year_end}')

    # collect slices of inputs from shared mem
    _input_array_sza = np.frombuffer(var_dict['input_array_sza']).reshape(var_dict['input_shape_sza'])[year_start:year_end, j, i]     # day, lat, lon
    _input_array_ghi = np.frombuffer(var_dict['input_array_ghi']).reshape(var_dict['input_shape_ghi'])[year_start:year_end, j, i]

    if noon == 'solar':

        # calculate solar noon
        #print(f'lon: {lon}, lat: {lat}, snoon: {snoon[0]}, {snoon[180]}')

        # calculate solar noon
        gamma = 2 * pi / 365 * np.arange(365)
        eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
        snoon =      (4 * _lon - eqtime) - tz[i] * 60
        snoon = pd.to_timedelta(snoon, unit='minutes')
        time = (var_dict['time'] + snoon).tz_localize(df.iloc[i, j], ambiguous=True, nonexistent='shift_backward')
    else:
        time = var_dict['time']
        time += pd.to_timedelta(tz[i], unit='hours')

    _output = np.empty((_time_dim, ))
    for d, day in enumerate(time):#[0:24]):
        # get variables for each day

        #print(day)

        _day_index = math.floor(d/24)

        _ghi = _input_array_ghi[d]
        _sza = _input_array_sza[d]

        _dni = pvlib.irradiance.disc(ghi=_ghi, solar_zenith=_sza, datetime_or_doy=day)
        _output[d] = _dni['dni']


    return (i, j, _output)


def ghi_decomp(input_array_ghi, input_array_sza, startYear, endYear):

    if noon in ('solar', 'clock'): # NEED TO CHANGE TO ALLOW FOR LEAP YEARS
        _time_dim = 365
    else:
        _time_dim = 365 * 24

    # read input data
    df = pd.read_csv('elev_tz.csv')
    df.columns = ['idx', 'lon', 'lat', 'alt', 'tz']
    tz = pd.read_csv('grid_with_tz.csv').zone.dropna().values

    lon = pd.DataFrame(df.lon.unique(), columns=['lon'])
    lon['id'] = range(1, len(lon.index) + 1)

    lat = pd.DataFrame(df.lat.unique(), columns=['lat'])
    lat['id'] = range(1, len(lat.index) + 1)

    df = df.merge(lat, on='lat').rename(columns={'id': 'lat_id'}).merge(lon, on='lon').rename(columns={'id': 'lon_id'})
    df = df.pivot(index='lon_id', columns='lat_id', values='tz')

    # load input arrays
    #input_array_ghi = np.load('ccsm_2020_hourly_ghi_UPDATED90.npy')#[0:24, :, :]
    #input_array_sza = np.load(f'sza_hourly_2020.npy').reshape(8760, 192, 288)#[0:24, :, :]

    # cache original shapes
    input_shape_ghi = input_array_ghi.shape
    input_shape_sza = input_array_sza.shape

    assert input_shape_ghi == input_shape_sza# == input_shape_pres

    # create shared input arrays
    raw_input_ghi = RawArray('d', input_shape_ghi[0] * input_shape_ghi[1] * input_shape_ghi[2])
    raw_input_sza = RawArray('d', input_shape_sza[0] * input_shape_sza[1] * input_shape_sza[2])

    assert len(raw_input_ghi) == len(raw_input_sza)# == len(raw_input_pres)

    # wrap inputs as numpy arrays
    input_np_ghi = np.frombuffer(raw_input_ghi).reshape(input_shape_ghi)

    input_np_sza = np.frombuffer(raw_input_sza).reshape(input_shape_sza)

    assert input_np_ghi.shape == input_np_sza.shape #== input_np_pres.shape

    # copy data to shared input array
    np.copyto(input_np_ghi, input_array_ghi)

    #logging.debug('copying SZA np array')
    np.copyto(input_np_sza, input_array_sza)

    assert input_array_ghi.shape == input_array_sza.shape #== input_array_pres.shape
    assert input_np_ghi.shape == input_np_sza.shape #== input_np_pres.shape

    for y, year in enumerate(range(startYear, endYear)):
        # set time range
        #logging.info(f'[{y}:{year}:{noon}] starting')
        if noon in ('solar', 'clock'):
            time = pd.date_range(start=f'{year}-01-01 12:00:00', end=f'{year}-12-31 12:00:00', freq='D')
        else:
            time = pd.date_range(start=f'{year}-01-01 00:00:00', end=f'{year}-12-31 23:00:00', freq='H')

        if calendar.isleap(year):
            time = time[:int(-_time_dim/365)]#[~((time.day == 29) & (time.month == 2))]

        assert time.shape[0] == _time_dim

        _indices = [list(_) + [y, ] for _ in product(range(lon.shape[0]), range(lat.shape[0]))]

        assert input_array_ghi.shape == input_array_sza.shape #== input_array_pres.shape
        assert input_shape_ghi == input_shape_sza# == input_shape_pres

        with Pool(processes=12,
                  initializer=init_worker,
                  initargs=(input_array_ghi, input_array_sza,# input_array_pres,
                            input_shape_ghi, input_shape_sza,# input_shape_pres,
                            time)) as pool:

            results = pool.map(worker_func, _indices)

        _output = np.empty((_time_dim, 192, 288))
        for result in results:
            i, j, dni = result
            _output[:, j, i] = dni

        np.save(f'dni_{noon}_noon_{year}.npy', _output)
        #logging.info(f'[{y}:{year}:{noon}] complete')

    assert 0

    out = np.empty((_time_dim * 20, 192, 288))
    out.fill(np.nan)

    for y, year in enumerate(range(2020, 2040)):
        start_day = y * _time_dim
        end_day = start_day + _time_dim

        x = np.load(f'dni_{noon}_noon_{year}.npy')

        x = x.reshape((_time_dim, 192, 288))

        out[start_day:end_day, :, :] = x

    assert not np.sum(np.isnan(out))

    np.save(f'dni_{noon}_noon.npy', out.copy(order='C'))
    return out
    #logging.info('complete!')
