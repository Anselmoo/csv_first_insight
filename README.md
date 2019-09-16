[![Build Status](https://travis-ci.com/Anselmoo/csv_first_insight.svg?branch=master)](https://travis-ci.com/Anselmoo/csv_first_insight)
[![CodeFactor](https://www.codefactor.io/repository/github/anselmoo/csv_first_insight/badge)](https://www.codefactor.io/repository/github/anselmoo/csv_first_insight)
[![codebeat badge](https://codebeat.co/badges/7cfcb1a6-8106-4022-bf22-575f2171c1fc)](https://codebeat.co/projects/github-com-anselmoo-csv_first_insight-master)
[![Mergify Status](https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/Anselmoo/csv_first_insight&style=flat)](https://github.com/Anselmoo/csv_first_insight/commits/master)
[![DOI](https://zenodo.org/badge/207603311.svg)](https://zenodo.org/badge/latestdoi/207603311)
![GitHub](https://img.shields.io/github/license/Anselmoo/csv_first_insight)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/Anselmoo/csv_first_insight)](https://github.com/Anselmoo/csv_first_insight/releases)
# CSV-First-Insights
A [`sklearn`](https://scikit-learn.org/stable/index.html)-based *correlation- and prediction-maker* for small csv-data < 10,000 entries. Consquently, *no* Neural Network will be used and so far the following Models are implemented:

- [Ridge-Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Gradient-Boosting-Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [Random-Forest-Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

Furthermore, for a first analysis, the cluster- and *aprori*-pair-plots can be easily generated for checking dependencies in the data.

The **CSV-First-Insights**-application can be installed like this:

    python setup.py install
 
The options of the Command Line Interface of **CSV-First-Insights** are:

    python -m pyinsights --help
    usage: __main__.py [-h] [--fname FNAME FNAME] [--mode MODE] [--export]

    Analyzer for small (# < 10,000) csv-Databases with binary content via scikit-learn! 
    Training-Set and Test-Set is separately stored in two databases.

    optional arguments:
    -h, --help           show this help message and exit
    --fname FNAME FNAME  Two filenames have to be defined for the train- and test-set. 
                         Default names are: train-data.csv','test-data.csv'
    --mode MODE          Please chose the model for the forecaset: 
                          *Ridge-Regression as a Variation of Linear-Regressions -> rig(deafault) 
                          *Gradient-Boosting-Trees -> grad 
                          *Random-Forest -> fors 
                          *All three models, please choose -> all
    --export             Export the Apriori-Analysis, Cluster-Maps, and Predictions as png- and txt-file

The **CSV-First-Insights** can be also loaded as packages like this:
```python
import pyinsights
import pyinsights.dataread as dr
import pyinsights.mlmodels as ml
import pyinsights.sklsetups as skl
```

The Ridge-Regression-Prediction of **CSV-First-Insights** for the [The Ultimate Halloween Candy Power Ranking](https://www.kaggle.com/fivethirtyeight/the-ultimate-halloween-candy-power-ranking) of kaggle:

![](https://github.com/Anselmoo/csv_first_insight/blob/master/docs/DecissionBar_ridge_reg_prediction.png)
