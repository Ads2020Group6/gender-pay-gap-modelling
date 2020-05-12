import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


def kfold_eval_all_models():
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
                'EmpSize5k', 'EmpSizeLt250', 'SectorAgriculture', 'SectorMining',
                'SectorConstruction', 'SectorManufacturing', 'SectorUtilityServices',
                'SectorWholesaleTrade', 'SectorRetailTrade', 'SectorFinancials',
                'SectorServices', 'SectorPublicAdministration', 'SectorNonclassifiable',
                'FirstSicCodeAsNum'
                ]

    df = pd.read_csv('data/ukgov-gpg-all-clean-with-features.csv')
    df = df.dropna(axis=0, subset=features)  # droping missing values everywhere

    X = df[features]
    kf = KFold(n_splits=3, random_state=42, shuffle=True)

    models = dict(
        linear_reg=LinearRegression(normalize=True),
        decision_tree=DecisionTreeRegressor(random_state=1, max_depth=10, min_samples_split=10),
        adaboost=AdaBoostRegressor(random_state=1),
        gradboost=GradientBoostingRegressor(random_state=1),
        random_forest=RandomForestRegressor(random_state=1),
        extra_trees=ExtraTreesRegressor(bootstrap=False,
                                        max_features=0.7500000000000001,
                                        min_samples_leaf=1,
                                        min_samples_split=2,
                                        n_estimators=100),
        extra_trees2=ExtraTreesRegressor(bootstrap=True, max_features=0.6000000000000001,
                                         min_samples_leaf=1,
                                         min_samples_split=9,
                                         n_estimators=100),
        xgboost=XGBRegressor(max_depth=9,
                             learning_rate=0.013,
                             n_estimators=2000,
                             silent=True,
                             nthread=-1,
                             gamma=0,
                             min_child_weight=1,
                             max_delta_step=0,
                             subsample=0.75,
                             colsample_bytree=0.85,
                             colsample_bylevel=1,
                             reg_alpha=0,
                             reg_lambda=1,
                             scale_pos_weight=1,
                             seed=1440,
                             missing=None)
    )

    for target in ['DiffMeanHourlyPercent', 'DiffMedianHourlyPercent']:
        y = df[target]
        fold_idx = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            for name, model in models.items():
                print(name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                yield target, fold_idx, name, r2, mae, mse
            fold_idx += 1


def main():
    df = pd.DataFrame(columns=('prediction', 'kFoldIndex', 'modelName', 'r2', 'MeanAveErr', 'MeanSqErr'))
    for target, fold_idx, name, r2, mae, mse in kfold_eval_all_models():
        result = dict(prediction=target,
                      kFoldIndex=fold_idx,
                      modelName=name,
                      r2=r2,
                      MeanAveErr=mae,
                      MeanSqErr=mse)
        print(result)
        df = df.append(result,
                       ignore_index=True)
    df.to_csv('data/model_run_results.csv', index=False)


if __name__ == "__main__":
    main()
