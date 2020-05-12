"""
Download data from original sources if they are not already present in the data dir
"""

import argparse
import os
from pathlib import Path

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
        print("{} already exists in {}".format(local_filename, target_dir))
        return

    print("Downloading {} to {}".format(local_filename, target_dir))
    with requests.get(url, stream=True) as r:
        with test_path.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)

    return local_filename


def main():
    parser = argparse.ArgumentParser(description='Download the UK Gender Pay Gap data and associated data files')
    parser.add_argument('--overwrite', type=bool, default=False,
                        help='whether to overwrite existing files (default is to not download again)')

    args = parser.parse_args()

    if args.overwrite:
        for year in (2017, 2018, 2019):
            delete_file('data', 'ukgov-gpg-{}.csv'.format(year))
        delete_file('data', 'SIC07_CH_condensed_list_en.csv')

    for year in (2017, 2018, 2019):
        download_file_if_not_exist(
            url='https://gender-pay-gap.service.gov.uk/viewing/download-data/{}'.format(year),
            target_dir='data',
            filename="ukgov-gpg-{}.csv".format(year))

    SIC_CODES_CSV='https://github.com/nathanpitman/sic-codes/blob/master/2007/sic_codes.csv'
    download_file_if_not_exist(
        url=SIC_CODES_CSV,
        target_dir='data',
        filename='sic_codes.csv'
    )


if __name__ == "__main__":
    main()
