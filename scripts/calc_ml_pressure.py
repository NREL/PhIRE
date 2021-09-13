import cdsapi
from pathlib import Path
import xarray as xa


def download(path):
    path = Path(path)
    if not path.exists():
        c = cdsapi.Client()
        c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': 'surface_pressure',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
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
        'time': [
            '00:00', '02:00',
            '04:00', '06:00', 
            '08:00', '10:00', 
            '12:00', '14:00', 
            '16:00', '18:00', 
            '20:00', '22:00',
        ],
        'grid':'0.75/0.75',
        'year': '2020',
    },
    str(path))

if __name__ == '__main__':
    download('/data/ERA5/single-level/surface_pressure_2020.grib')
    ds = xa.open_dataset('/data/ERA5/single-level/surface_pressure_2020.grib', engine='cfgrib')

    a_119 = 3057.265625
    b_119 = 0.873929

    a_121 = 2294.242188
    b_121 = 0.899900

    pf_120 = (a_119 + a_121) / 2 + (b_119 + b_121) * ds.sp / 2
    pf_120 = pf_120 / 100 # [Pa] -> [hPa]

    mean, std = float(pf_120.mean()), float(pf_120.std())
    print(f'{mean:.2f}+-{std:.2f} [hPa]')