import pandas as pd
import numpy as np
import re


def drop_dupes(df):
    df.drop_duplicates(inplace=True)
    return df


def impute_missing_mean_and_median_vals(df):
    # Mean because underlying statistic is mean
    mean_bonus_percent = df['DiffMeanBonusPercent'].mean()
    df['DiffMeanBonusPercent'] = df['DiffMeanBonusPercent'].fillna(mean_bonus_percent)
    mean_hourly_percent = df['DiffMeanHourlyPercent'].mean()
    df['DiffMeanHourlyPercent'] = df['DiffMeanHourlyPercent'].fillna(mean_hourly_percent)

    # Median because the measurement is median
    median_bonus_percent = df['DiffMedianBonusPercent'].median()
    df['DiffMedianBonusPercent'] = df['DiffMedianBonusPercent'].fillna(median_bonus_percent)
    median_hourly_percent = df['DiffMedianHourlyPercent'].mean()
    df['DiffMedianHourlyPercent'] = df['DiffMedianHourlyPercent'].fillna(median_hourly_percent)
    return df


def merge_years(df2017, df2018, df2019):
    df2017['year'] = 2017
    df2018['year'] = 2018
    df2019['year'] = 2019
    df = pd.concat([df2017, df2018, df2019])
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


def one_hot_enc_company_size(df):
    one_hot = pd.get_dummies(df['EmployerSize']).rename(
        columns={"Less than 250": "EmpSizeLt250",
                 "250 to 499": "EmpSize250",
                 "500 to 999": "EmpSize500",
                 "1000 to 4999": "EmpSize1k",
                 "5000 to 19,999": "EmpSize5k",
                 "20,000 or more": "EmpSize20k"})

    df = df.merge(one_hot, left_index=True, right_index=True)
    return df


def numerical_company_size(df):
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


def sic_as_num(df):
    def first_sic(sics):
        tmp = str(sics).split('\r\n')
        tmp = [t.strip() for t in tmp]
        tmp = [re.sub(',', '', t) for t in tmp]
        tmp = [np.float(x) / 100 if x != 1 else -1 for x in tmp]
        return tmp[0] if len(tmp) > 0 else -1

    df['FirstSicCodeAsNum'] = df.SicCodes.map(first_sic)
    return df


def hot_enc_toplevel_sic_sector(df):
    # See https://en.wikipedia.org/wiki/Standard_Industrial_Classification
    mappings = {
        'Agriculture': ('01', '09'),
        'Mining': ('10', '14'),
        'Construction': ('15', '17'),
        'Manufacturing': ('20', '39'),
        'UtilityServices': ('40', '49'),
        'WholesaleTrade': ('50', '51'),
        'RetailTrade': ('52', '59'),
        'Financials': ('60', '69'),
        'Services': ('70', '90'),
        'PublicAdministration': ('91', '97'),
        'Nonclassifiable': ('98', '99')
    }

    def clean_sic(sics):
        tmp = str(sics).split('\r\n')
        tmp = [t.strip() for t in tmp]
        tmp = [re.sub(',', '', t) for t in tmp]
        return tmp

    df['CleanedUpSic'] = df.SicCodes.map(clean_sic)

    def contains_range(sector, list_of_sics):
        for sic in list_of_sics:
            from_code, to_code = mappings[sector]
            if from_code <= sic[:2] <= to_code:
                return 1.
        return 0.

    for key in mappings.keys():
        df['Sector{}'.format(key)] = df.CleanedUpSic.map(lambda x: contains_range(key, x))

    df.drop('CleanedUpSic', axis=1, inplace=True)
    return df


def drop_no_sicdata(df):
    df = df.dropna(axis=0, subset=['SicCodes'])
    return df


def main():
    # df_sic = pd.read_csv('data/SIC07_CH_condensed_list_en.csv')
    df_2017 = pd.read_csv('data/ukgov-gpg-2017.csv', dtype={'SicCodes': str})
    df_2018 = pd.read_csv('data/ukgov-gpg-2018.csv', dtype={'SicCodes': str})
    df_2019 = pd.read_csv('data/ukgov-gpg-2019.csv', dtype={'SicCodes': str})
    df = merge_years(df_2017, df_2018, df_2019)
    del df_2017
    del df_2018
    del df_2019

    df = drop_dupes(df)
    df = drop_unused_cols(df)
    df = impute_missing_mean_and_median_vals(df)
    df = drop_where_numerical_feature_is_na(df)
    df = drop_no_sicdata(df)
    df = numerical_company_size(df)
    df = one_hot_enc_company_size(df)
    df = hot_enc_toplevel_sic_sector(df)
    df = sic_as_num(df)
    df.drop('SicCodes', axis=1, inplace=True)

    df.to_csv('data/ukgov-gpg-all-cleaned.csv', index=False )


if __name__ == "__main__":
    main()
