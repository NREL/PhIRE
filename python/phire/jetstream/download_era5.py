import cdsapi
from multiprocessing import Pool

def download_month(year, month, client):
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': 'u_component_of_wind',
            'pressure_level': [
                '700', '775', '850',
                '925',
            ],
            'year': str(year),
            'month': f'{month:02d}',
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'area': [
                75, -60, 15,
                0,
            ],
        },
        f'u_wind_{year}_m{month}.grib')


def download_year(year):
    with Pool(6) as pool:
        futures = []
        for month in range(1, 13):
            c = cdsapi.Client()
            futures.append(pool.apply_async(download_month, (year, month, c)))
        
        for f in futures:
            f.wait()

def main():
    for year in range(1981, 2021):
        download_year(year)


if __name__ == '__main__':
    main()