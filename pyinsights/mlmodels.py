# Own buildin-functions
# External functions
import argparse

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate, RandomizedSearchCV


from pyinsights import *


class PlotModes(object):

    def __init__(self):
        self.coefficient = []
        self.ordering = []
        self.zer = np.empty([10])
        self.inc = np.empty([10])
        self.res = np.empty([10])

    def plot_models(self, title="Ridge regression feature coefficients", fname='DecissionBar.png', plot=False):
        """
        Plotting the most important attributes
        str :param fname:
        :param plot:
        img :return: as png-file
        """
        plt.clf()  # cleaning previous plot because of the seanborn libary for avoiding typo-crash
        sns.set(style="whitegrid")

        sns.set(font_scale=0.6)

        # Generating correlation-bar-stick-graph
        plt.figure(figsize=(5, 7))

        elm = skl.elements(el=self.ordering)
        plt.bar(elm, self.coefficient)
        plt.xticks(elm, self.ordering, rotation=35)
        plt.title(title)

        plt.savefig(fname, dpi=300)
        if plot: plt.show()

    def plot_residium(self, fname='ResiduumResults_PredRes.png', plot=False):
        """
        Plotting the residuum of the predictions
        str :param fname:
        :param plot:
        img :return: as png-file
        """
        plt.clf()  # cleaning previous plot because of the seanborn libary for avoiding typo-crash
        sns.set(style="whitegrid")
        plt.figure(figsize=(5, 7))
        plt.plot(self.inc, self.res, 'o-')
        plt.plot(self.inc, self.zer, '--')
        plt.xlim([min(self.inc), max(self.inc)])
        plt.xlabel("# of Elements")
        plt.ylabel("$\Delta$ between reference and predict")
        plt.title("Residuum between reference and fitted predictor")

        plt.savefig(fname, dpi=300)
        if plot: plt.show()


class PredictModel(PlotModes):
    def __init__(self):
        self.train = pd.DataFrame()
        self.features = []
        self.brandname = []
        self.predicts = []
        self.zer = np.empty([10])
        self.inc = np.empty([10])
        self.res = np.empty([10])

    def protoype(self, fname):
        data = dr.data_read(fname=fname, winper=False)
        self.features, self.train = skl.get_x_train(data=data, index_name=['competitorname'])
        self.brandname = data['competitorname']

    def predict(self, reg):
        self.predicts = reg.predict(self.train)

    def run(self, fname, reg):
        """
        Function for predicting the sucess of new candy-bars
        str :param fname: *.csv for predicting new candy-bars
        class :param reg: scklearn-model
        list :return: prediction coefficient of the wineprercent
        """
        self.protoype(fname=fname)
        self.predict(reg=reg)
        return self.predicts, self.brandname

    def verify(self, train, reg, ref, fname='ResiduumResults_PredRes.png'):
        """
        To figure out the numerical-accuracy of the prediction
        class:param ref: scklearn-prediction as a reference
        class:param reg: scklearn-model
        list :return: prediction coefficient of the wineprercent
        """
        self.train = train
        self.predict(reg=reg)
        self.zer = np.zeros_like(ref, dtype=float)  # Set-up the empty residuum-array
        self.inc = np.arange(len(self.zer))
        self.res = np.subtract(ref, self.predicts)

        self.plot_residium(fname=fname)

        return self.res


class MlModels(PredictModel):
    """
    This class contains the following machine-learning methods:
    *Linear Regression in the ridge-regression fashion
    *Regression tree as Gradient Boosting Trees
    *RandomForestRegressor
    """

    def ridge_reg(self, X, y, features):
        """
        2D-array :param X: feature of the candy bars as pandas-type 2D-array (numpy.ndarray)
        1D-array :param y: winepercent as pandas-type 1D-array
        1D-array :param features: str-array to select the columns in panda-data
        1D-array, 1D-array, class :return: prediction coefficients as 1D-array, order of the attributes, reg as skilearn-class


        More about the Ridge-Regression (a Variation of  Linear-Regressions):
        https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db
        O'Reilly - Hands-On Machine Learning with Scikit-Learn & Tensorflow p. 129ff.
        """

        # Fit a series of various linear-regression to find the best alpha
        # alphas increment from 0.05 to 5.00 in a fixed 0.05 step
        ridgeReg = RidgeCV(alphas=skl.alphas(), scoring='neg_mean_squared_error', cv=None)
        ridgeReg.fit(X=X, y=y)
        print("Best Alpha ->", ridgeReg.alpha_)
        scores = cross_validate(ridgeReg, X=X, y=y, cv=X.shape[0], scoring='neg_mean_squared_error')
        print("Baseline linear regression mean squared error (MSE) score ->", abs(np.mean(scores['test_score'])))

        # Indexing and ordering from most to less important attributes
        index = np.argsort(-ridgeReg.coef_)
        ordering = skl.feature_sort(features=features, index=index)

        self.verify(train=X, reg=ridgeReg, ref=y, fname='ResiduumResults_PredRes.png')
        return ridgeReg.coef_[index], ordering, ridgeReg

    def grad_tree(self, X, y, features):
        """
        2D-array :param X: feature of the candy bars as pandas-type 2D-array (numpy.ndarray)
        1D-array :param y: winepercent as pandas-type 1D-array
        1D-array :param features: str-array to select the columns in panda-data
        1D-array, 1D-array, class :return: prediction coefficients as 1D-array, order of the attributes, reg as skilearn-class


        More about the Gradient Boosting Trees:
        https://astro.temple.edu/~msobel/courses_files/StochasticBoosting(gradient).pdf
        O'Reilly - Hands-On Machine Learning with Scikit-Learn & Tensorflow p. 169ff.
        """

        gbmReg = GradientBoostingRegressor(loss='ls', criterion='friedman_mse')
        paramlist = {'max_depth': range(1, 15), 'n_estimators': np.arange(10, 61, 1),
                     'learning_rate': np.logspace(0, -3, 40, base=10)}

        n_iter = 200
        # This loop is for iterating the best set of parameters
        random_search = RandomizedSearchCV(gbmReg, scoring='neg_mean_squared_error', param_distributions=paramlist,
                                           n_iter=n_iter, cv=X.shape[0] // 10, iid=True)

        random_search.fit(X=X, y=y)
        print("Gradient Boosting Trees best parameters ->", random_search.best_params_)
        print("Gradient Boosting Trees mean squared error (MSE) score ->", abs(random_search.best_score_))

        # Indexing and ordering from most to less important attributes
        index = np.argsort(-random_search.best_estimator_.feature_importances_)
        ordering = skl.feature_sort(features=features, index=index)

        self.verify(train=X, reg=random_search, ref=y, fname='ResiduumResults_GradientTree.png')

        return random_search.best_estimator_.feature_importances_[index], ordering, random_search

    def random_forest(self, X, y, features):
        """
        2D-array :param X: feature of the candy bars as pandas-type 2D-array (numpy.ndarray)
        1D-array :param y: winepercent as pandas-type 1D-array
        1D-array :param features: str-array to select the columns in panda-data
        1D-array, 1D-array, class :return: prediction coefficients as 1D-array, order of the attributes, reg as skilearn-class


        More about the Random-Forest:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        O'Reilly - Hands-On Machine Learning with Scikit-Learn & Tensorflow p. 191ff.
        """

        randfors = RandomForestRegressor(criterion='mse')

        paramlist = {'max_depth': range(1, 15), 'n_estimators': np.arange(10, 61, 1),
                     'random_state': np.linspace(10, 200, 30, dtype=int),
                     'min_samples_leaf': np.arange(1, 3)}

        n_iter = 200
        # This loop is for iterating the best set of parameters
        random_search = RandomizedSearchCV(randfors, scoring='neg_mean_squared_error', param_distributions=paramlist,
                                           n_iter=n_iter, cv=X.shape[0] // 10, iid=True)

        random_search.fit(X=X, y=y)
        print("Random Forest Regression best parameters ->", random_search.best_params_)
        print("Random Forest Regression mean squared error (MSE) score ->", abs(random_search.best_score_))

        # Indexing and ordering from most to less important attributes
        index = np.argsort(-random_search.best_estimator_.feature_importances_)
        ordering = skl.feature_sort(features=features, index=index)

        self.verify(train=X, reg=random_search, ref=y, fname='ResiduumResults_RandomForest.png')

        return random_search.best_estimator_.feature_importances_[index], ordering, random_search


class TrainSet(MlModels, PredictModel, PlotModes):

    def __init__(self):
        super().__init__()
        self.data = pd.DataFrame()
        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.features = []
        self.coefficient = []
        self.ordering = []
        self.plot = False
        self.fname = ''

    def initialize(self, fname, export=False):
        """
        Reading the candy_bar_reference *csv-file
        :param fname: filename for csv
        :param export: exporting plots and analysis
        :return:
        """
        self.data = dr.data_read(fname=fname)
        if export:
            # calling the export functions
            dr.data_plot(data=self.data)
            dr.data_apri(data=self.data)
            dr.data_corl(data=self.data)

    def trainset_split(self):
        """

        self.x_train :return: 2D-array with characteristics for best candy-bar (numpy.ndarray)
        self.y_train :return: 1D-array with win per candy bar

        """
        self.features, self.x_train = skl.get_x_train(data=self.data, index_name=['competitorname', 'winpercent'])
        self.y_train = self.data['winpercent']

    def run_models(self, mode='linear'):
        # Loading the MlModels-class
        if mode == 'linear':
            self.fname = 'DecissionBar_ridge_reg'
            self.coefficient, self.ordering, self.reg = self.ridge_reg(X=self.x_train, y=self.y_train,
                                                                       features=self.features)


        elif mode == 'tree':
            self.fname = 'DecissionBar_grad_tree'
            self.coefficient, self.ordering, self.reg = self.random_forest(X=self.x_train, y=self.y_train,
                                                                           features=self.features)

        elif mode == 'forest':
            self.fname = 'DecissionBar_random_forest'
            self.coefficient, self.ordering, self.reg = self.grad_tree(X=self.x_train, y=self.y_train,
                                                                       features=self.features)

        self.plot_models(fname=self.fname + '_prediction.png')

    def predict_models(self, fname):
        # Fname of the test-reference-file
        predicts, brandname = self.run(fname=fname, reg=self.reg)

        # Indexing and ordering from most to less important attributes
        index = np.argsort(-predicts)  # As an index-list for the brandname
        self.coefficient = np.abs(np.sort(-predicts))  # As float-list for the prediction
        # as absolute-value because of the inverser sorting

        self.ordering = skl.feature_sort(features=brandname, index=index)

        self.plot_models(fname=self.fname + '_NewCandyBar.png', title="New candybars-prediction")

    def __del__(self):
        # That's optional to remove the garbage from the data- and prediction-variables
        self.data
        self.x_train
        self.y_train


if __name__ == "__main__":
    """
    Here later an argphraser will control the engine
    """

    despription_init = "Analyzer of the Candy-Bar-Databas. infile for database.\n\tThe infile-name of the database " \
                       "has to be: candy-data.csv\n\tThe infile-name of suggestion-databese has to be: candy-prototypes.csv'"
    parser = argparse.ArgumentParser(description=despription_init)
    parser.add_argument('--export', help="Export the Apriori-Analysis as well "
                                         "as the Cluster-Maps as png file to get the correlation", action='store_true')
    parser.add_argument("--mode", type=str, default='rig', help="Please chose the model for the forecaset:\n"
                                                                "\t*Ridge-Regression as a Variation of  Linear-Regressions "
                                                                "-> rig (deafault)\n"
                                                                "\t*Gradient-Boosting-Trees -> grad\n"
                                                                "\t*Random-Forest -> fors\n"
                                                                "\nIf you are planning to use all three models, "
                                                                "please choose -> all"
                        )
    parser.add_argument("--fname", type=str, default='candy-prototypes.csv', help="Filename of the candy-bar-prototypes! Default name is"
                                                                                  "candy-prototypes.csv")
    args = parser.parse_args()
    ts = TrainSet()
    ts.initialize(fname='candy-data.csv', export=args.export)
    ts.trainset_split()

    try:
        """
        Here the comannd line input will be checked for the model. In case of all three models are required
        we need three if checks, otherwise it will stop at the first if-elif-else statement
        Finally, a checker for choosen the wrong or not listed model.
        """
        if args.mode.lower() == 'rig' or args.mode.lower() == 'all':
            print("Ridge-Regression-Method is chosen!")
            ts.run_models(mode='linear')  # Learning
            ts.predict_models(fname=args.fname)  # Testing
            print("Done!")
        if args.mode.lower() == 'grad' or args.mode.lower() == 'all':
            print("Gradient-Boosting-Trees-Method is chosen!")
            ts.run_models(mode='tree')  # Learning
            ts.predict_models(fname=args.fname)  # Testing
            print("Done!")
        if args.mode.lower() == 'fors' or args.mode.lower() == 'all':
            print("Random-Forest-Method is chosen!")
            ts.run_models(mode='forest')  # Learning
            ts.predict_models(fname=args.fname)  # Testing
            print("Done!")
        if not ['rig', 'grad', 'fors', 'all'].__contains__(args.mode.lower()):
            print("This mode is not available!")

    except:
        print("Here is an error, please write me: Anselm.Hahn@icloud.com")


