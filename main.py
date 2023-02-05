import argparse
import json
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

from dimreduction import apply_dimreduction, perform_MCA, perform_PCA
from feature_shuffling import feature_shuffling
from forests import run_randomforests

from sklearn.naive_bayes import MultinomialNB, ComplementNB, CategoricalNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from class_limiter import class_limit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


np.random.seed(42)
random.seed(69)


RESERVED_ATTRIBUTES = ['key', 'is-design', 'is-cat1', 'is-cat2', 'is-cat3']
ONE_HOT_ATTRIBUTES = ['labels', 'status',  'resolution', 'issuetype']


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='data.json', type=str,
                        help='Specify the input file with data to use')
    parser.add_argument('--model', default='RF', type=str,
                        help='Specify the model to use: NB, DT, RF')
    parser.add_argument('--output_type', default='detection', type=str,
                        help='Specify the output mode: detection or classification')
    parser.add_argument('--dimred', default='PCA', type=str,
                        help='Specify the dimension reduction method: PCA or MCA')
    parser.add_argument('--feature_shuffling', action='store_true',
                        help='Perform feature shuffling benchmark')
    parser.add_argument('--class_limit', action='store_true',
                        help='Enables class limiting')
    parser.add_argument('--confusion_matrix', action='store_true',
                        help='Enable generation of confusion matrices')
    args = parser.parse_args()
    return args

dim = 49

with open("features.txt") as f:
    feature_names = f.read().splitlines()
    

def get_features(args):
    X = []
    Y = []
    with open(args.data_file) as file:
        data = json.load(file)

        for item in data:
            metadata = []
            if item['resolution' ] is None:
                continue
            for key, value in item.items():
                if key in RESERVED_ATTRIBUTES:
                    continue
                if key in ONE_HOT_ATTRIBUTES:
                    metadata.extend(value)
                else:
                    metadata.append(value)
            if len(metadata) != dim:
                continue
            X.append(metadata)
            if args.output_type == 'detection':
                Y.append(item['is-design'])
            elif args.output_type == 'classification':
                if item['is-cat2']['value']:
                    Y.append(np.array([1, 0, 0, 0]))
                elif item['is-cat3']['value']:
                    Y.append(np.array([0, 1, 0, 0]))
                elif item['is-cat1']['value']:
                    Y.append(np.array([0, 0, 1, 0]))
                else:
                    Y.append(np.array([0, 0, 0, 1]))
            else:
                raise 'specify a correct output_type'

    X_np = np.zeros([len(X), len(X[0])])
    for idx in range(len(X)):
        X_np[idx, :] = X[idx]

    if args.output_type == 'classification':
        Y = [np.argmax(item) for item in Y]

    
    return X_np, np.array(Y)

def create_model(args, weights):
    if args.model == 'NB':
        return ComplementNB(
            alpha=0.5,
            fit_prior=False,
            force_alpha=True
        )
    elif args.model == 'DT':
        if args.output_type == 'detection':
            return DecisionTreeClassifier(
                class_weight=weights,
                criterion='entropy',
                max_depth=5,
                max_features=None,
                splitter='random'
            )
        return DecisionTreeClassifier(
            class_weight=weights,
            criterion='log_loss',
            max_depth=5,
            max_features=None,
            splitter='best'
        )
    elif args.model == 'RF':
        if args.output_type == 'detection':
            return RandomForestClassifier(
                class_weight=weights,
                criterion='entropy',
                max_depth=10,
                max_features='log2',
                n_estimators=50
            )
        return RandomForestClassifier(
            class_weight=weights,
            criterion='entropy',
            max_depth=10,
            max_features='log2',
            n_estimators=200
        )
    else:
        raise 'Specify a valid model'

def create_distribution(model, weights):
    if model == 'NB':
        distributions = dict(
            alpha=[0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
            force_alpha=[True],
            fit_prior=[False, True]
        )
    elif model == 'DT':
        distributions = dict(
            criterion=['gini', 'entropy', 'log_loss'],
            splitter=['best', 'random'],
            max_depth=[5, 10, 15, 20, 25],
            max_features=[None, 'sqrt', 'log2'],
            class_weight=[weights]
        )
    elif model == 'RF':
        distributions = dict(
            n_estimators=[50, 100, 200, 500],
            criterion=['gini', 'entropy', 'log_loss'],
            max_depth=[5, 10, 15, 20, 25],
            max_features=[None, 'sqrt', 'log2'],
            class_weight=[weights]
        )

    return distributions


def main():
    args = create_argparser()
    X, Y = get_features(args)

    if args.class_limit:
        X, Y = class_limit(X, Y)

    weights = {key: len(Y) / value for key, value in Counter(Y).items()}

    model = create_model(args, weights)



    if args.feature_shuffling:
        feature_shuffling(model, X, Y, args)
        return
    distributions = create_distribution(args.model, weights)

    #apply_dimreduction(X, args.dimred, 10)

    # forests = run_randomforests(X,Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)

    search = GridSearchCV(model, distributions, scoring='f1_macro',
                         verbose=2, n_jobs=6)
    search.fit(X_train, Y_train)

    

    # Show the results
    print('Best hyperparameters:', search.best_params_)
    print('Accuracy of best hyperparameters:', search.best_score_)
    # Write cv results to text file
    with open("gridsearch_cvresults_rf.txt", 'w') as f:
        f.write('Best hyperparameters: %s\n' % (search.best_params_))
        f.write('Accuracy of best hyperparameters: %s\n' % (search.best_score_))
        for key, value in search.cv_results_.items():
            f.write('%s:%s\n' % (key, value))

    model.fit(X_train, Y_train)
    if args.model == 'DT' and args.output_type == 'detection':
        tree.export_graphviz(model, out_file='DT_detection.dot', rounded=True, filled=False, feature_names=feature_names,
        class_names=['Non-Architectural', 'Architectural'],)
    elif args.model == 'DT' and args.output_type == 'classification':
        tree.export_graphviz(model, out_file='DT_classification.dot', rounded=True, filled=False, feature_names=feature_names,
        class_names=['Executive', 'Property', 'Existence', 'Non-Architectural'])

    Y_pred = model.predict(X_test)
    print('Test set performance (f-score):', f1_score(Y_test, Y_pred, average='macro'))

    if args.output_type == 'detection':
        cm_labels = ['Non-Architectural', 'Architectural']
    else:
        cm_labels = ['Executive', 'Property', 'Existence', 'Non-Architectural']
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    disp.plot()
    plt.show()

    #var_array = np.array([perform_PCA(X, i) for i in range(dim)])
    #print(var_array)


if __name__ == '__main__':
    main()