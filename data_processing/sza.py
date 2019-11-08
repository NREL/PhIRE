import numpy as np
import pandas as pd

#data = np.load('sza_hourly_2020_OLD.npy')
#print(np.min(data), np.mean(data), np.max(data))
#exit()

def sza(latitude, lon_num, days_in_year):
    '''
        latitude - an array of all of the unique latitudes in the region
        lon_num - should be the the number of longitudes we will be doing the decomposition over.
        days_in_year - either 365 or 366
    '''

    print(latitude.shape)
    lats = np.zeros((latitude.shape[0] * lon_num))
    for i,j in enumerate(latitude):
        lats[i:i + lon_num] = j
        print(lats[i:i + lon_num].shape)
    print(lats.shape)

    lat_rads = lats*(np.pi/180)

    time_steps = 24 * days_in_year #should be either 8760 or 8784
    sza = np.zeros((time_steps, lats.shape[0]))
    for i, lat in enumerate(lat_rads):
        if (i % 1000 == 0):
            print(i, 'of ', lats.shape[0])
        j = 0
        for day in range(365):
            declination = 23.45*(np.pi/180)*np.sin(2*np.pi*((284+day)/365.25))
            for hour in range(1, 25):
                w = (hour - 12.5)*(np.pi/12)
                sza[j, i] = np.arccos(np.sin(lat)*np.sin(declination) + np.cos(lat)*np.cos(declination)*np.cos(w))
                j += 1
    #np.save('sza_hourly_2020_2.npy', sza)
    return sza

if __name__ == '__main__':

    l = np.load("ccsm_all_2020-2039_lats_only_avg.npy")
    t = sza(l, 288)
    print(t.shape, np.min(t), np.mean(t), np.max(t))
