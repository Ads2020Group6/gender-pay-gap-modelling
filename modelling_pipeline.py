import pandas as pd
from joblib import dump, load
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path
from models import models
from math import sqrt

features = ['MaleBonusPercent', 'FemaleBonusPercent',
            'MaleLowerQuartile', 'FemaleLowerQuartile',
            'MaleLowerMiddleQuartile', 'FemaleLowerMiddleQuartile',
            'MaleUpperMiddleQuartile', 'FemaleUpperMiddleQuartile',
            'MaleTopQuartile', 'FemaleTopQuartile',
            'BonusGenderSkew', 'WorkforceGenderSkew',
            'RepresentationInLowerMiddleQuartileSkew',
            'RepresentationInUpperMiddleQuartileSkew',
            'RepresentationInLowerQuartileSkew',
            'RepresentationInTopQuartileSkew',
            'PercMaleWorkforceInTopQuartile', 'PercMaleWorkforceInUpperMiddleQuartile',
            'PercMaleWorkforceInLowerMiddleQuartile', 'PercMaleWorkforceInLowerQuartile',
            'PercFemaleWorkforceInTopQuartile', 'PercFemaleWorkforceInUpperMiddleQuartile',
            'PercFemaleWorkforceInLowerMiddleQuartile', 'PercFemaleWorkforceInLowerQuartile',
            'year', 'EmployerSizeAsNum', 'EmpSize1k', 'EmpSize20k', 'EmpSize250', 'EmpSize500',
            'EmpSize5k', 'EmpSizeLt250',
            'SectA', 'SectB', 'SectC', 'SectD', 'SectE',
            'SectF', 'SectG', 'SectH', 'SectI', 'SectJ', 'SectK',
            'SectL', 'SectM', 'SectN', 'SectO', 'SectP', 'SectQ',
            'SectR', 'SectS', 'SectT', 'SectU',
            'FirstSicCodeAsNum'
            ]


def split_holdout_companies(df):
    # We want to withhold some companies (all years!) from the data entirely, to use
    # for cross validation
    companies = df['CompanyNumber'].unique()
    test_train, validate = train_test_split(companies, test_size=0.1, shuffle=True)
    df['holdout'] = df['CompanyNumber'].isin(validate)

    X_val = df[df['holdout']][features]
    y_val = df[df['holdout']][['DiffMeanHourlyPercent', 'DiffMedianHourlyPercent']]
    X = df[~df['holdout']][features]
    y = df[~df['holdout']][['DiffMeanHourlyPercent', 'DiffMedianHourlyPercent']]

    #save to file for checking predictions
    holdout = df[df['holdout']]
    holdout.to_csv('data/complete_holdout_data.csv')
    return X, y, X_val, y_val


def kfold_eval_all_models(X, targets):
    kf = KFold(n_splits=3, random_state=42, shuffle=True)

    for target in ['DiffMeanHourlyPercent', 'DiffMedianHourlyPercent']:
        y = targets
        fold_idx = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index][target].values, y.iloc[test_index][target].values
            for name, model in models.items():
                print(name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                yield target, fold_idx, name, r2, mae, sqrt(mse)
            fold_idx += 1


def train_and_pickle_best_model(bestModelName, target, X, y):
    print('Retraining {} for {} on full (non-holdout) set'.format(bestModelName, target))
    model = models[bestModelName]
    model.fit(X, y)
    dump(model, 'models/{}-best-model.joblib'.format(target))


def evaluate_best_model_on_holdout(target, X_val, y_val):
    print("---")
    print("Evaluating model for {} on holdout data".format(target))
    print("---")
    model = load('models/{}-best-model.joblib'.format(target))
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    print("R^2:   {:.2}".format(r2))
    print("MAE:   {:.2}".format(mae))
    print("RMSE:   {:.2}".format(sqrt(mse)))


def main(retrain=True, pickle=True):
    df = pd.read_csv('data/ukgov-gpg-all-clean-with-features.csv')
    df = df.dropna(axis=0, subset=features)  # droping missing values everywhere

    print('Splitting of 10% of companies (all years) as holdout data')
    X, y, X_val, y_val = split_holdout_companies(df)
    holdout = X_val.merge(y_val, left_index=True, right_index=True)
    holdout.to_csv('data/holdout_data.csv')
    print('Holdout data written to data/holdout.csv')

    if retrain:
        print('Evaluating all models in 3-fold validation of test_train data')
        print('Warning: this takes a long time!')
        df = pd.DataFrame(columns=('prediction', 'kFoldIndex', 'modelName', 'r2', 'MeanAveErr', 'RootMeanSqErr'))
        for target, fold_idx, name, r2, mae, rmse in kfold_eval_all_models(X, y):
            result = dict(prediction=target,
                          kFoldIndex=fold_idx,
                          modelName=name,
                          r2=r2,
                          MeanAveErr=mae,
                          RootMeanSqErr=rmse)
            print(result)
            df = df.append(result,
                           ignore_index=True)
        df.to_csv('data/model_run_results.csv', index=False)

    df = pd.read_csv('data/model_run_results.csv')
    grouped = df.groupby(['modelName', 'prediction']).mean().reset_index()
    grouped.to_csv('data/averaged_model_scores.csv', index=False)
    best_model_mean = grouped[grouped['prediction'] == 'DiffMeanHourlyPercent'].loc[
        grouped[grouped['prediction'] == 'DiffMeanHourlyPercent']['r2'].idxmax()]
    best_model_median = grouped[grouped['prediction'] == 'DiffMedianHourlyPercent'].loc[
        grouped[grouped['prediction'] == 'DiffMedianHourlyPercent']['r2'].idxmax()]
    print(best_model_mean[['modelName', 'r2', 'MeanAveErr', 'RootMeanSqErr']])
    print(best_model_median[['modelName', 'r2', 'MeanAveErr', 'RootMeanSqErr']])

    if pickle:
        # Now that we know the best for Mean and Median, train
        # on the whole dataset (without holdout validation) and pickle models
        Path('models').mkdir(parents=True, exist_ok=True)
        train_and_pickle_best_model(best_model_mean['modelName'], best_model_mean['prediction'], X,
                                    y['DiffMeanHourlyPercent'].values)
        train_and_pickle_best_model(best_model_median['modelName'], best_model_median['prediction'], X,
                                    y['DiffMedianHourlyPercent'].values)

    evaluate_best_model_on_holdout('DiffMeanHourlyPercent', X_val, y_val['DiffMeanHourlyPercent'])
    evaluate_best_model_on_holdout('DiffMedianHourlyPercent', X_val, y_val['DiffMedianHourlyPercent'])


if __name__ == "__main__":
    print('Warning! This takes a long time...')
    main(retrain=True, pickle=True)


