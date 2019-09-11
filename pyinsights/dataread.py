__all__ = ['data_read','data_corl','data_plot','data_apri']
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



def data_plot(data, fname=['Correlation.png','ClusterMap.png'], plot=False):
    """
    Compute pairwise correlation of columns and print as png by using the searbon-lib
    2D-obj :param data: pandas-data-object (numpy.ndarray)
    str  :param fname: filename for the png of the correlation-graph
    bool :param plot: show the plot via tinker-interface
    img  :return: png-file
    """



    sns.set(font_scale=0.6)

    correlation = data.corr() #Pandas correlation-routine
    # Generating correlation-map
    plt.figure(figsize=(13, 13))

    sns.heatmap(correlation, center=0, vmin=-1, vmax=1,
                   square=True, annot=True, cmap='bwr', fmt='.1f',
                   linewidths=.75)

    plt.title('Hierarchically-clustering between different features')
    plt.savefig(fname[0], dpi=300)

    # Generating clustermap
    sns.clustermap(correlation, center=0, vmin=-1, vmax=1,
                   square=True, annot=True, cmap='bwr', fmt='.1f',
                   linewidths=.75, figsize=(13, 13))


    plt.title('Correlation between different features')
    plt.savefig(fname[1], dpi=300)
    if plot: plt.show()


def data_apri(data, fname='apriori.png', plot=False):
    """
    2D-obj :param data: pandas-data-object (numpy.ndarray)
    str  :param fname: filename for the png of the apriori-graph
    bool :param plot: show the plot via tinker-interface
    img :return:  png-file

    The apriori-algorithm based on:
    Agrawal, Rakesh, and Ramakrishnan Srikant. "Fast algorithms for mining association rules."
    Proc. 20th int. conf. very large data bases, VLDB. Vol. 1215. 1994.
    """
    winners = data[data.winpercent > data.winpercent.quantile(.6)]

    data_f = winners[data.columns[1:-3]]
    association = apriori(data_f, min_support=0.3, use_colnames=True).sort_values(by='support')
    plt.figure(figsize=(9,7))
    association.plot(kind='barh', x='itemsets', y='support', title=f'Most Frequently Used Composition',
                     sort_columns=True, figsize=(10, 5), legend=True)
    plt.savefig(fname, dpi=300)
    if plot:
        plt.show()


if __name__ == "__main__":
    data = data_read(fname='candy-data.csv',norm=['winpercent'])

    def printf(data):
        print(data.head())
        print(data.columns.values)

    printf(data)
    #data_plot(data)
    #data_corl(data)
    #data_apri(data)
