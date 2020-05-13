import data_collector
import data_cleaner
from augment_features import augment 
from modelling_pipeline import features 
from joblib import dump, load
from sector_exploder import explode_sectors
import numpy as np
import pandas as pd
def clean_up(df):
    df = data_cleaner.drop_dupes(df)
    df = data_cleaner.drop_unused_cols(df)
    df = data_cleaner.drop_where_numerical_feature_is_na(df)
   # df = impute_missing_mean_and_median_vals(df)
    df = data_cleaner.drop_no_sicdata(df)
    df = data_cleaner.numerical_company_size(df)
    df = data_cleaner.one_hot_enc_company_size(df)
    df = data_cleaner.hot_enc_toplevel_sic_sector(df)
    df = data_cleaner.sic_as_num(df)
    df.drop('SicCodes', axis=1, inplace=True)
    # df = explode_sectors(df)

    return df


def main():
    # get data
    # year = 2019
    # data_collector.download_file_if_not_exist(
    #     url='https://gender-pay-gap.service.gov.uk/viewing/download-data/{}'.format(
    #         year),
    #     target_dir='data',
    #     filename="ukgov-gpg-{}.csv".format(year))
    df_2019 = pd.read_csv('data/ukgov-gpg-2019.csv', dtype={'SicCodes': str})
    df = df_2019.copy()
    # clean up
    df_2019 = clean_up(df_2019)
    # feature engineering
    df_2019 = augment(df_2019)
    df_2019['year'] = 2019
    X = df_2019[features]
    
    #get model
    mean_model = load('models/{}-best-model.joblib'.format('DiffMeanHourlyPercent'))
    median_model = load('models/{}-best-model.joblib'.format('DiffMedianHourlyPercent'))
    
    # #predict
    mean_predictions = mean_model.predict(X)
    median_predictions = median_model.predict(X)

    df_2019['error'] = np.abs(median_predictions - df_2019['DiffMedianHourlyPercent'])
    printcols = ['EmployerName','CompanyNumber','DiffMeanHourlyPercent','DiffMedianHourlyPercent','error','CompanyLinkToGPGInfo']
    print(df_2019[printcols].sort_values('error',ascending=False).head(10))


if __name__ == "__main__":
    main()
