__all__ = ['data_head','data_read','data_corl','data_plot','data_apri']
"""
The initial data-csv read and export functions:
The function-set consist of:

data_read(): -> converting csv into a datastream in the pandas-style
data_plot(): -> generating the correlation-graph between the features
data_corl(): -> generating the correlation between the features as an ascii-export (*txt)
data_apri(): -> generating an apriori-analysis on the flight
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from mlxtend.frequent_patterns import apriori

def data_head(fname):
    """
    Get the columns-names of the csv

    Parameters
    ----------
    fname: str
        Filename of the csv-data

    Returns
    ----------

    str-list:
        header-names of the csv-data
    """
    return pd.read_csv(fname, encoding='ISO-8859-1').columns

def data_read(fname, norm=None):
    """
    Reads the csv-file and exports it to pandas.object.
    Optional it normalize columns by dividing by float(100)

    Parameters
    ----------

    fname: str
        Filename of the csv-data
    norm: str-list
        Columns-Names for 100%-division

    Returns
    ----------

    data: ndarray
        pandas.dataframe with string header

    """


    data = pd.read_csv(fname, encoding='ISO-8859-1')
    if norm is not None:
        for hname in norm:
            data[hname] /= 100.
    return data

def data_corl(data, fname=['abs_Correlation.csv', 'Correlation.csv']):
    """
    Make a combination list of all features meaning as an absolute list
    Make a combination list of all features meaning as an relative list


    Parameters
    ----------

    data: ndarray
        pandas.dataframe with string header
    fname: str-list
        Filename of the to exporting csv-data

    Returns
    ----------
        csv: data
            Save to database with absolute and relative errors of the input-dataset

    """
    df_1 = pd.DataFrame([[i, j, data.corr().abs().loc[i, j]] for i, j in list(combinations(data.corr().abs(), 2))]
                        , columns=['Feature1', 'Feature2', 'abs(corr)'])  # Generating all combinations
    df_1.sort_values(by='abs(corr)', ascending=False).reset_index(drop=True)

    df_2 = pd.DataFrame([[i, j, data.corr().loc[i, j]] for i, j in list(combinations(data.corr(), 2))]
                        , columns=['Feature1', 'Feature2', 'corr'])  # Generating all combinations
    df_2.sort_values(by='corr', ascending=False).reset_index(drop=True)

    df_1.to_csv(fname[0], index=True)
    df_2.to_csv(fname[1], index=True)

def data_plot(data, fname=['Correlation.png','ClusterMap.png'], plot=False, save=False):
    """
    Compute pairwise correlation of columns and print as png by using the searbon-lib

    Parameters
    ----------
    data: ndarray
        pandas.dataframe with string header
    fname: str-list
        Filename of the to exporting csv-data
    plot: bool
        switch on plt.show()
    save: bool
        save the matplotlib-plot as png
    """



    sns.set(font_scale=0.6)

    correlation = data.corr() #Pandas correlation-routine
    # Generating correlation-map
    #plt.figure(figsize=(13, 13))

    sns.heatmap(correlation, center=0, vmin=-1, vmax=1,
                   square=True, annot=True, cmap='bwr', fmt='.1f',
                   linewidths=.75)

    plt.title('Hierarchically-clustering between different features')
    if save:
        plt.savefig(fname[0], dpi=300)

    # Generating clustermap
    sns.clustermap(correlation, center=0, vmin=-1, vmax=1,
                   square=True, annot=True, cmap='bwr', fmt='.1f',
                   linewidths=.75)


    plt.title('Correlation between different features')
    if save:
        plt.savefig(fname[1], dpi=300)
    if plot:
        plt.show()

def data_apri(data, keyel, keybl=[1,-3], fname='apriori.png', threshold=0.6, plot=False,save=False):
    """
    A quick apriori-analysis to find correlating pairs

    Notes
    -----
    The apriori-algorithm based on:
    Agrawal, Rakesh, and Ramakrishnan Srikant. "Fast algorithms for mining association rules."
    Proc. 20th int. conf. very large data bases, VLDB. Vol. 1215. 1994.

    See Also
    --------
    https://en.wikipedia.org/wiki/Apriori_algorithm

   Parameters
    ----------
    data: ndarray
        pandas.dataframe with string header
    keyel: str
        column-name for  the critical key-element
    keybl: int-list
        list for the array-indices for the poor binaries (0-1) entries in numpy-notation
    fname: str-list
        Filename of the to exporting csv-data
    threshold: float
        lower threshold for the quick comparsion
    plot: bool
        switch on plt.show()
    save: bool
        save the matplotlib-plot as png
    """
    winners = data[data[keyel] > data[keyel].quantile(threshold)]

    data_f = winners[data.columns[keybl[0]:keybl[1]]]
    association = apriori(data_f, min_support=0.3, use_colnames=True).sort_values(by='support')
    #plt.figure(figsize=(9,7))
    association.plot(kind='barh', x='itemsets', y='support', title=f'Most Frequently Used Composition',
                     sort_columns=True, figsize=(10, 5), legend=True)
    if save:
        plt.savefig(fname, dpi=300)
    if plot:
        plt.show()

if __name__ == "__main__":
    data = data_read(fname='train-data.csv',norm=['winpercent'])

    def printf(data):
        print(data.head())
        print(data.columns.values)

    printf(data)
    data_plot(data)
    data_corl(data)
    data_apri(data,keyel='winpercent')
