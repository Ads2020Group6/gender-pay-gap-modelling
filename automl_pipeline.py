from math import sqrt

import pandas as pd
from joblib import dump
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from tpot import TPOTRegressor

from modelling_pipeline import features, split_holdout_companies


def train_and_pickle_best_model(target, X, y, val_X, val_y):
    print('AutoML Search for good model for {}'.format(target))
    pipeline_optimizer = TPOTRegressor(generations=10, population_size=150, cv=3,
                                       random_state=0xDEADBEEF, verbosity=3, scoring='r2',
                                       n_jobs=-1, early_stop=5, periodic_checkpoint_folder='tpot_checkpoint')
    pipeline_optimizer.fit(X, y)
    new_preds = pipeline_optimizer.predict(val_X)
    mae = mean_absolute_error(val_y, new_preds)
    rmse = sqrt(mean_squared_error(val_y, new_preds))
    r2 = r2_score(val_y, new_preds)
    print("TPOT mae:", mae)
    print("TPOT rmse:", rmse)
    print("TPOT R^2 score:", r2)
    pipeline_optimizer.export('models/tpot_exported_pipeline_{}.py'.format(target))
    dump(pipeline_optimizer.fitted_pipeline_, 'models/{}-best-model-automl.joblib'.format(target))
    return r2, mae, rmse


def main(retrain=True, pickle=True):
    df = pd.read_csv('data/ukgov-gpg-all-clean-with-features.csv')
    df = df.dropna(axis=0, subset=features)  # dropping missing values everywhere

    print('Splitting of 10% of companies (all years) as holdout data')
    X, y, X_val, y_val = split_holdout_companies(df)
    holdout = X_val.merge(y_val, left_index=True, right_index=True)
    holdout.to_csv('data/holdout_data_automl.csv')
    print('Holdout data written to data/holdout_data_automl.csv')

    for target in ['DiffMeanHourlyPercent', 'DiffMedianHourlyPercent']:
        X_, y_ = shuffle(X, y[target].values, random_state=42)
        df = pd.DataFrame(columns=('prediction', 'r2', 'MeanAveErr', 'MeanSqErr'))
        r2, mae, rmse = train_and_pickle_best_model(target, X_, y_, X_val, y_val[target].values)
        result = dict(prediction=target,
                      r2=r2,
                      MeanAveErr=mae,
                      RootMeanSqErr=rmse)
        df = df.append(result,
                       ignore_index=True)
    df.to_csv('data/automl_model_results.csv', index=False)


if __name__ == "__main__":
    print('Warning! This takes a VERY long time... (5hrs?)')
    main()
