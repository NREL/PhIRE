''' @author: Andrew Glaws, Karen Stengel, Dylan Hettinger
'''
import numpy as np
import matplotlib.pyplot as plt

def angle_TAG_hour(hour):
    if (hour >= 1 and hour <= 24):
        dw = np.pi/12
        start = (hour - 13)*dw #w1
        middle = (hour - 12.5)*dw #w
        end = (hour - 12)*dw #w2
    else:
        print("invalid hour value : ", hour)
    return start, middle, end

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
                sza[j, i] = np.arccos(np.abs(np.sin(lat)*np.sin(declination) + np.cos(lat)*np.cos(declination)*np.cos(w)))
                j += 1
    #np.save('sza_hourly_2020_2.npy', sza)
    return sza

def clearnessIndex(useTAGeq, hour, GHI):
    if (hour >= 1 and hour <= 24):

        if useTAGeq is True:
            # use general equation from TAG paper:
            Kt_h = 0.88*np.cos(np.pi*(hour - 12.5)/30)
        else:
            #actually calculate the ratio
            Kt_h = GHI/1361 #from dylan MIGHT CHANGE TO CALCULATE ACTUAL VALUE

    else:
        print("invalid hour value : ", hour)

    return Kt_h

def getGHI(Kt_h, extratrest):
    #MIGHT CHANGE TO ACCEPT ALL DAYS BUT TBD
    return Kt_h*extratrest

def riseSet(declination, latitude):
    cos_angle = -np.tan(latitude)*np.tan(declination)
    angle = np.where(cos_angle<=1, np.where(cos_angle>=-1, np.arccos(cos_angle), np.pi), 0)

    return angle

def plotHourlyKt(kt, kt_avg, mask, unconverged_mask):
    vmin_inst, vmax_inst = np.min(kt), np.max(kt)
    vmin_avg, vmax_avg = np.min(kt_avg), np.max(kt_avg)
    vmin, vmax = np.minimum(vmin_inst, vmin_avg), np.maximum(vmax_inst, vmax_avg)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    ret = (ret[n - 1:] / n)
    ret = np.concatenate(([a[0]], ret, [a[-1]]), axis=0)

    return ret

def genHourClearness(phi, mask, kt_cs, kt_m, sig, y, kt_avg):
    kt = np.zeros_like(mask)
    unconverged_mask = np.zeros_like(mask[0, ...], dtype=np.bool)

    valid, iters = False, 0
    while(valid is False):
        bad_range = np.zeros_like(mask)

        for hour in range(1, 25):
            sig_prime = sig[hour-1, ...]*np.sqrt(1 - phi**2)
            sig_prime = sig_prime[unconverged_mask]
            r = np.random.normal(scale=sig_prime, size=sig_prime.shape)

            y_hold = y[hour-1, ...].copy()
            if hour == 1:
                y_tmp = r
            else:
                y_prev_hold = y[hour-2, ...].copy()
                y_tmp = phi[unconverged_mask]*y_prev_hold[unconverged_mask] + r
            y_hold[unconverged_mask] = y_tmp
            y[hour-1, ...] = y_hold


            kt_hold = kt[hour-1, ...].copy()
            kt_tmp = mask[hour-1, ...]*(kt_m[hour-1, ...] + sig[hour-1, ...]*y[hour-1, ...])
            kt_hold[unconverged_mask] = kt_tmp[unconverged_mask]
            kt[hour-1, ...] = kt_hold

            bad_range[hour-1, ...] = (kt[hour-1, ...] < 0) | np.all(kt[hour-1, ...] > kt_cs[hour-1])

        kt_avg_syn = np.sum(mask*kt, axis=0)/np.sum(mask, axis=0)
        rel_err = np.where(kt_avg > 0, np.abs(kt_avg_syn - kt_avg)/kt_avg, kt_avg_syn)

        check_range = np.any(bad_range, axis=0)
        check_err = (rel_err > 0.03)
        unconverged_mask = check_range | check_err

        if not np.any(unconverged_mask) or (iters > 50):
            valid = True
            print(iters, ' - Convegence percent:', 1 - np.sum(unconverged_mask)/float(unconverged_mask.size))
        else:
            iters += 1

    return kt

def tag_alg(kt_avg_obs, nsrdb_ghi_avg, latitude, day_shift=1):
    lat_rads = latitude * (np.pi/180)

    kt_hr_syn_all = None
    for day in range(kt_avg_obs.shape[0]):
        print('Day:', day)
        kt_avg_obs_day = kt_avg_obs[day, :, :]

        kt_cs = np.zeros((24, ))
        y = np.zeros((24, kt_avg_obs.shape[1], kt_avg_obs.shape[2]))
        sig = np.zeros((24, kt_avg_obs.shape[1], kt_avg_obs.shape[2]))
        kt_m = np.zeros((24, kt_avg_obs.shape[1], kt_avg_obs.shape[2]))
        kt_hr_syn = np.zeros((24, kt_avg_obs.shape[1], kt_avg_obs.shape[2]))
        mask = np.zeros((24, kt_avg_obs.shape[1], kt_avg_obs.shape[2]))
        smooth_mask = np.zeros_like(mask)

        declination = 23.45*(np.pi/180)*np.sin(2*np.pi*((284+day+day_shift)/365.25))
        wss = riseSet(declination, lat_rads)

        for hour in range(1, 25):
            w1, w, w2 = angle_TAG_hour(hour)

            mask[hour-1, ...] = (w2 >= -wss) & (w1 <= wss)
            tmp_mask = mask[hour-1, :, 0]
            for _ in range(15):
                tmp_mask = moving_average(tmp_mask, n=3)
            smooth_mask[hour-1, ...] = np.tile(tmp_mask.reshape((-1, 1)), (1, mask.shape[2]))

        kt_avg_obs_day = kt_avg_obs_day*24/np.sum(smooth_mask, axis=0)

        phi = 0.38 + 0.06*np.cos(7.4*kt_avg_obs_day - 2.5)

        for hour in range(1, 25):
            #print(day, hour)

            #STEP 1
            w1, w, w2 = angle_TAG_hour(hour)
            sin_hs = np.sin(lat_rads)*np.sin(declination) + np.cos(lat_rads)*np.cos(declination)*np.cos(w)
            sin_hs[sin_hs <= 1e-8] = 1e-8

            if np.any(smooth_mask[hour-1, ...] > 0):#np.any(mask[hour-1, ...]):

                # STEP 2
                kt_cs[hour-1] = clearnessIndex(True, hour, nsrdb_ghi_avg[day, :, :])

                #STEP 3
                e = 0.32 - 1.60*(kt_avg_obs_day - 0.5)**2
                k = 0.19 + 2.27*kt_avg_obs_day**2 - 2.51*kt_avg_obs_day**3
                l = -0.19 + 1.12*kt_avg_obs_day + 0.24*np.exp(-8*kt_avg_obs_day)
                kt_m[hour-1, ...] = l + e*np.exp(-k/sin_hs)

                #STEP 4
                A = 0.14*np.exp(-20*(kt_avg_obs_day - 0.35)**2)
                B = 3*(kt_avg_obs_day - 0.45)**2 + 16*kt_avg_obs_day**5
                sig[hour-1, ...] = A*np.exp(B*(1 - sin_hs))

        kt_hr_syn = genHourClearness(phi, smooth_mask, kt_cs, kt_m, sig, y, kt_avg_obs_day)
        if kt_hr_syn_all is None:
            kt_hr_syn_all = kt_hr_syn
        else:
            kt_hr_syn_all = np.concatenate((kt_hr_syn_all, kt_hr_syn), axis=0)

    return kt_hr_syn_all
