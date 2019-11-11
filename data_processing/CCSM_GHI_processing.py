from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tag import *
from sza import *
#from ghi_decomp_by_year import *

#DEFAULTS
theoretical_GHI_limit = 1361

def load_nc_file(file_name):
    # var is going to be the main variable
    #lat is the lat in an array as the same size as var
    #lon is the lons in an array as the same size as var
    fh_rsds = Dataset(file_name, mode = "r")
    lat = fh_rsds.variables['lat'][:]
    lon = fh_rsds.variables['lon'][:]
    var = fh_rsds.variables['rsds'][:]
    return var, np.reshape(lat, (lat.shape[0], 1)), np.reshape(lon, (lon.shape[0], 1))


if __name__ == '__main__':

    startYear = 2020
    endYear = 2021
    #CHANGE TO TAKE IN NC FILE AND EXTRACT EVERYTHING NEEDED FROM THERE.
    nsrdb_ghi_avg, latitude, longitude = load_nc_file("rsds_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc")
    print(nsrdb_ghi_avg.shape, latitude.shape, longitude.shape)

    latitude_2d = np.broadcast_to(latitude, (latitude.shape[0], longitude.shape[0]))

    print("hi ", latitude_2d)
    #latitude = latitude[32:160, :]

    kt_avg_obs = nsrdb_ghi_avg/theoretical_GHI_limit
    print(np.mean(kt_avg_obs))

    kt_hfa = tag_alg(kt_avg_obs, nsrdb_ghi_avg, latitude_2d, 0)
    kt_hfa = np.where(kt_hfa >= 0, kt_hfa, 0)
    print(kt_hfa.shape)
    print(np.min(kt_hfa), np.mean(kt_hfa), np.max(kt_hfa))
    print('')

    hourly_ghi = getGHI(kt_hfa, 1361)
    print(hourly_ghi.shape)
    print(np.min(hourly_ghi), np.mean(hourly_ghi), np.max(hourly_ghi))

    np.save('ccsm_2020_hourly_ghi.npy', hourly_ghi)

    #GET GENERAL SZA
    tag_sza = sza(latitude, 288, days)
    #CALL THE DISC MODEL. NEED TO CLEAN UP A LOT
    dni = ghi_decomp(hourl_ghi, tag_sza, startYear, endYear)
    dhi = 0
