__all__ = ['feature_sort','alphas','elements', 'get_X_train']

import numpy as np
def feature_sort(features, index):
    """
    Sorts the features according the index-list
    
    Parameters
    ----------
    features: list
        with str, int, float, boal elements
    index: int-list 
    for sorting/ re-ordering
    
    Returns
    ----------
    
    A sorted feature-list       
    """
    return [features[i] for i in index]

def alphas(start=0.05,stop=5.00,step=0.05):
    """
    Return evenly spaced values within a given interval with fixed thresholf of 0.001 minimum stepsize
 
    Parameters
    ----------
    start : number, optional
        Start of interval
            
    stop : number
        End of interval
    step : number
        Incrimiment of the interval
       
    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.
    """
    if start >= 0.001 and stop >= 0.001 and step >= 0.001:
        return np.arange(start=start, stop=stop, step=step)
    else:
        print("Warning input guess is illegal!\n")
        print("Return: start=0.05,stop=5.0,step=0.05")
        return np.arange(start=0.05,stop=5.0,step=0.05)

def elements(el):
    """
    Return evenly spaced values starting from 0 within a given interval of 1 for the maximum numbers of elements (el)
    """
    return np.arange(len(el))

def get_X_train(data, index_name):
    """
    Import the pandas-data and select and sliced it according the column-index-list

    Parameters
    ----------

    data: ndarray
        Pandas data-stream
    index_name: str-list
        Columns-Names for removing columns out the data-stream

    Returns
    ----------

    features: str-list
        feature-list of the not-selected column-names
    X_train: ndarray
        pandas.dataframe for X-dataset

    """

    features = [i for i in list(data.columns.values) if i not in index_name]
    X_train = data[features]
    return features, X_train

if __name__ == "__main__":
    print(alphas(start=-0.001,stop=0.002,step=0.001))


