"""
Download data from original sources if they are not already present in the data dir
"""

import argparse
import os
from pathlib import Path
import pandas as pd

import requests


def delete_file(target_dir, filename):
    test_path = Path(os.path.join(target_dir, filename))
    if test_path.is_file():
        os.remove(test_path)


# Download the Gender pay gap data from UK Gov if it's not already there
def download_file_if_not_exist(url, target_dir='data', extension='', filename=None):
    local_filename = filename if filename is not None else url.split('/')[-1] + extension

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    test_path = Path(os.path.join(target_dir, local_filename))
    if test_path.is_file():
        print("{} already exists in '{}' folder".format(local_filename, target_dir))
        return

    print("Downloading {} to {}".format(local_filename, target_dir))
    with requests.get(url, stream=True) as r:
        with test_path.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)

    return local_filename

def download_data():
    for year in (2017, 2018, 2019):
        download_file_if_not_exist(
            url='https://gender-pay-gap.service.gov.uk/viewing/download-data/{}'.format(year),
            target_dir='data',
            filename="ukgov-gpg-{}.csv".format(year))

    SIC_CODES_CSV='https://raw.githubusercontent.com/nathanpitman/sic-codes/master/2007/sic_codes.csv'
    download_file_if_not_exist(
        url=SIC_CODES_CSV,
        target_dir='data',
        filename='sic_codes.csv'
    )

def merge_years(df2017, df2018, df2019):
    df2017['year'] = 2017
    df2018['year'] = 2018
    df2019['year'] = 2019
    return pd.concat([df2017, df2018, df2019])

def acquire_data(save_file=False, output_filename='data/ukgov-gpg-full.csv'):
    download_data()
    df_2017 = pd.read_csv('data/ukgov-gpg-2017.csv', dtype={'SicCodes': str})
    df_2018 = pd.read_csv('data/ukgov-gpg-2018.csv', dtype={'SicCodes': str})
    df_2019 = pd.read_csv('data/ukgov-gpg-2019.csv', dtype={'SicCodes': str})
    df_full = merge_years(df_2017, df_2018, df_2019)
    if save_file: df_full.to_csv(output_filename, index=False)
    return 

def main():
    parser = argparse.ArgumentParser(description='Download the UK Gender Pay Gap data and associated data files')
    parser.add_argument('--overwrite', type=bool, default=False,
                        help='whether to overwrite existing files (default is to not download again)')
    args = parser.parse_args()

    if args.overwrite:
        for year in (2017, 2018, 2019):
            delete_file('data', 'ukgov-gpg-{}.csv'.format(year))
        delete_file('data', 'sic_codes.csv')
    download_data()

if __name__ == "__main__":
    main()
