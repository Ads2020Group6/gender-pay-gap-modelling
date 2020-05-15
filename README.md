# Exploring the Gender Pay Gap

This repository contains the source code implemented for the modelling section of the Applied
Data Science unit at University of Bristol.

### Requirements
- Python 3+
- Jupyter (Notebook)
- Pandas
- Scikit-learn
- xgboost

### Files
- `data_collector`: Fetches the datasets from the UK government website.
- `data_cleaner`: Runs a number of cleaning and transformations on the data.
- `augment_features`: Adds gender representation to the dataset.
- `modelling_pipeline`: performs the training and evaluation of all tested.
- `automl_pipeline`
- `prediction`
The rest of the files are helpers or subcomponents.

### Run
To execute our complete pipeline run
```
> jupyter notebook
```
click `complete_pipeline.ipynb` and execute all cells.

### Update datasets
To pull and overwrite the existing datasets.
```
python data_collector.py --overwrite
```

### The Team
- Shivangi Das
- D. Leandro Guardia Vaca
- Thorben Louw
- Sarath Varman
- Kunlin Yang
- Binqian Zhu

### License
MIT License - Copyright (c) 2020 ads-group-6-overleaf

