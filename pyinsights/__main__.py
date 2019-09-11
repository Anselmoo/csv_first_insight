import argparse
from pyinsights.mlmodels import TrainSet

def cmd():
    """
    Here later an argphraser will control the engine
    """

    despription_init = "Analyzer for small (# < 10,000) csv-Databases with binary content via scikit-learn! " \
                       "Training-Set and Test-Set is separately stored in two databases."
    parser = argparse.ArgumentParser(description=despription_init)
    parser.add_argument("--fname", nargs=2, type=str, default=['train-data.csv', 'test-data.csv'],
                        help="Two filenames have to be defined for the train- and "
                             "test-set. Default names are: "
                             "train-data.csv','test-data.csv'")
    parser.add_argument("--mode", type=str, default='rig', help="Please chose the model for the forecaset:\n"
                                                                "\t*Ridge-Regression as a Variation of  Linear-Regressions "
                                                                "-> rig (deafault)\n"
                                                                "\t*Gradient-Boosting-Trees -> grad\n"
                                                                "\t*Random-Forest -> fors\n"
                                                                "\nIf you are planning to use all three models, "
                                                                "please choose -> all"
                        )
    parser.add_argument('--export', help="Export the Apriori-Analysis, Cluster-Maps, and Predictions as png- "
                                         "and txt-file", action='store_true')
    args = parser.parse_args()
    ts = TrainSet()
    ts.initialize(fname=args.fname[0], export=args.export)
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
            ts.predict_models(fname=args.fname[1])  # Testing
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

    except Exception as e:
        print("Here is an error:\n\t",e)

if __name__ == '__main__':
    cmd()
