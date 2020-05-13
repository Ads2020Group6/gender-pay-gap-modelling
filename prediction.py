import data_collector
from data_cleaner import clean_data
from augment_features import augment 
from modelling_pipeline import features 
from joblib import dump, load
import numpy as np
import pandas as pd

def main():
    # get data
    year = 2019
    data_collector.download_file_if_not_exist(
        url='https://gender-pay-gap.service.gov.uk/viewing/download-data/{}'.format(
            year),
        target_dir='data',
        filename="ukgov-gpg-{}.csv".format(year))
    df = pd.read_csv('data/ukgov-gpg-{}.csv'.format(year), dtype={'SicCodes': str})
    # df_copy = df.copy()
    # clean up
    df = clean_data(df,industry_sections="split")
    # feature engineering
    df = augment(df)
    df['year'] = 2019

    X = df[features]
    print("Getting the best models ----")
    # get model
    mean_model = load('models/{}-best-model.joblib'.format('DiffMeanHourlyPercent'))
    median_model = load('models/{}-best-model.joblib'.format('DiffMedianHourlyPercent'))
    
    # predict
    print("Predicting ----")
    mean_predictions = mean_model.predict(X)
    median_predictions = median_model.predict(X)

    df['mean_error'] = np.abs(mean_predictions - df['DiffMeanHourlyPercent'])
    df['median_error'] = np.abs(median_predictions - df['DiffMedianHourlyPercent'])

    print('Top 10 companies where the model got maximum abosute error in median GPG')
    printcols = ['EmployerName','CompanyNumber','DiffMeanHourlyPercent','DiffMedianHourlyPercent','mean_error','median_error','CompanyLinkToGPGInfo']
    print(df[printcols].sort_values('median_error',ascending=False).head(10))
    df.to_csv('data/predictions{}.csv'.format(year))
    print("Predictions written to predictions{}.csv".format(year))
if __name__ == "__main__":
    main()
