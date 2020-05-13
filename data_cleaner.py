import pandas as pd
import numpy as np
import re
from sic_transformer import explode_sectors, split_sectors


def drop_dupes(df):
    df.drop_duplicates(inplace=True)
    return df


def drop_unused_cols(df):
    del_cols = ["ResponsiblePerson", "SubmittedAfterTheDeadline", "DueDate", "DateSubmitted"]
    df.drop(del_cols, axis=1, inplace=True)
    return df


def drop_where_numerical_feature_is_na(df):
    features = ['DiffMedianBonusPercent', 'DiffMeanBonusPercent', 'MaleBonusPercent',
                'FemaleBonusPercent', 'MaleLowerQuartile', 'FemaleLowerQuartile',
                'MaleLowerMiddleQuartile', 'FemaleLowerMiddleQuartile',
                'MaleUpperMiddleQuartile', 'FemaleUpperMiddleQuartile',
                'MaleTopQuartile', 'FemaleTopQuartile']
    df = df.dropna(axis=0, subset=features)
    return df


def impute_missing_mean_and_median_vals(df):
    # Mean because underlying statistic is mean
    df = df.copy()
    mean_bonus_percent = df['DiffMeanBonusPercent'].mean()
    df['DiffMeanBonusPercent'].fillna(mean_bonus_percent, inplace=True)
    mean_hourly_percent = df['DiffMeanHourlyPercent'].mean()
    df['DiffMeanHourlyPercent'].fillna(mean_hourly_percent, inplace=True)

    # Median because the measurement is median
    median_bonus_percent = df['DiffMedianBonusPercent'].median()
    df['DiffMedianBonusPercent'].fillna(median_bonus_percent, inplace=True)
    median_hourly_percent = df['DiffMedianHourlyPercent'].mean()
    df['DiffMedianHourlyPercent'].fillna(median_hourly_percent, inplace=True)
    return df


def quantizise_employer_size(df):
    df = df.dropna(axis=0, subset=['EmployerSize'])

    def mid_point_employer_size(size_text):
        lut = {"Less than 250": 125,
               "250 to 499": 375,
               "500 to 999": 750,
               "1000 to 4999": 3000,
               "5000 to 19,999": 12500,
               "20,000 or more": 30000
               }
        return lut[size_text] if size_text in lut else -1

    df['EmployerSizeAsNum'] = df.EmployerSize.map(mid_point_employer_size)
    df.drop(df[df.EmployerSizeAsNum == -1].index, inplace=True)
    return df


def one_hot_enc_employer_size(df):
    one_hot = pd.get_dummies(df['EmployerSize']).rename(
        columns={"Less than 250": "EmpSizeLt250",
                 "250 to 499": "EmpSize250",
                 "500 to 999": "EmpSize500",
                 "1000 to 4999": "EmpSize1k",
                 "5000 to 19,999": "EmpSize5k",
                 "20,000 or more": "EmpSize20k"})

    df = df.merge(one_hot, left_index=True, right_index=True)
    return df


def clean_data(df, industry_sections="explode", save_file=False, output_filename='ukgov-gpg-full-clean-sections.csv'):
    # Runs dataset through a series of cleaning and transormation procedures.

    # Parameters:
    #    industry_sections (str):
    #       - None: does not parse SicCodes
    #       - explode: creates new rows for companies with more than one element in SicCodes
    #       - split: keeps one company row and distributes sections in respective columns 

    # Returns:
    #    Cleaned dataset.

    df = df.copy()
    df = drop_dupes(df)
    df = drop_unused_cols(df)
    df = impute_missing_mean_and_median_vals(df)
    df = drop_where_numerical_feature_is_na(df)
    df = quantizise_employer_size(df)
    df = one_hot_enc_employer_size(df)
    if industry_sections == "explode": df = explode_sectors(df)
    if industry_sections == "split": df = split_sectors(df)
    if save_file: df.to_csv(output_filename, index=False)
    return df


def main():
    # TODO: Argparser , input_filename, save_file, output_filename
    input_filename = "data/ukgov-gpg-full.csv"
    df = pd.read_csv(input_filename)
    return clean_data(df, industry_sections="split", save_file=True,
                      output_filename='data/ukgov-gpg-full-section-split.csv')


if __name__ == "__main__":
    df = main()
