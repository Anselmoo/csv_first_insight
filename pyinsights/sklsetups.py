__all__ = ['feature_sort','alphas','elements','get_x_train']

import numpy as np
def feature_sort(features, index):
    """

    1D-list :param features: with str, int, float, boal
    1D-list :param index: int-list for sorting/ ordering
    :return:
    """
    return [features[i] for i in index]

def alphas(start=0.05,stop=5.00,step=0.05):

    if start >= 0.001 and stop >= 0.001 and step >= 0.001:
        return np.arange(start=start, stop=stop, step=step)
    else:
        print("Warning input guess is illegal!\n")
        print("Return: start=0.05,stop=5.0,step=0.05")
        return np.arange(start=0.05,stop=5.0,step=0.05)

def elements(el):
    return np.arange(len(el))

def get_x_train(data,index_name=['competitorname', 'winpercent']):
    """
    2D-pandas-obj :param data: pandas-obs (numpy.ndarray)
    1D-list :param index_name: expresions to exculde
    1D-list, 2D-pandas :return: feature-list and 2D-pandas-object without the index_names
    """
    features = [i for i in list(data.columns.values) if i not in index_name]
    x_train = data[features]
    return features, x_train

if __name__ == "__main__":
    print(alphas(start=-0.001,stop=0.002,step=0.001))


