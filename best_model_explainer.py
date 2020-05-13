import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots
import shap
from pathlib import Path
from modelling_pipeline import features
from joblib import dump, load

plt.rcParams['figure.dpi'] = 150


def correlatePredictions(model, val_X, val_y, name):
    preds = model.predict(val_X)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(preds, val_y)
    # ax.plot(val_X.index, preds,label='Predicted value')
    # ax.legend()
    ax.set(xlabel='Predictions', ylabel='Actual pay gap',
           title=name)
    plt.savefig('viz/{}-correlation.png'.format(name), dpi=150)
    plt.show()



def feature_importance(model, X, target_name):
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['importance']).sort_values('importance', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(ax=ax, data=feature_importances.reset_index(), x='importance', y='index', orient='h')
    plt.savefig('viz/{}-fi.png'.format(target_name), dpi=150)
    plt.show()


def permutation_importance(model, X, y, target_name):
    perm = PermutationImportance(model, random_state=1).fit(X, y)
    eli5.show_weights(perm, feature_names=X.columns.tolist())
    plt.savefig('viz/{}-perm-imp.png'.format(target_name), dpi=150)
    plt.show()


def pdpFunction(model, data, feature_names, feature_to_plot, target_name):
    pdp_score = pdp.pdp_isolate(model=model, dataset=data, model_features=feature_names,
                                feature=feature_to_plot)
    pdp.pdp_plot(pdp_score, feature_to_plot)
    pdp.plt.savefig('viz/{}-{}-shap.png'.format(target_name, feature_to_plot), dpi=150)
    plt.show()


def plot_shap_values(model, X, target_name):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    # calculate shap values. This is what we will plot.
    # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    plt.savefig('viz/{}-shap.png'.format(target_name), dpi=150)


def explain_model(target_name, X, y):
    model = load('models/{}-best-model.joblib'.format(target_name))
    correlatePredictions(model, X, y, target_name)
    feature_importance(model, X, target_name)
    # plot_shap_values(model, X, target_name)

def main():
    Path('viz').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv('data/holdout_data.csv')
    X = df[features]
    for target in ['DiffMeanHourlyPercent', 'DiffMedianHourlyPercent']:
        y = df[target].values
        explain_model(target, X, y)


if __name__ == "__main__":
    main()
