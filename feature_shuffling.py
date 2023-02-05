import numpy as np
import copy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Store the property names and their feature lengths
property_lengths = [
    ['priority', 1],
    ['resolution', 4],
    ['status', 3],
    ['issuetype', 7],
    ['labels', 18],
    ['resolution_time', 1],
    ['n_components', 1],
    ['n_labels', 1],
    ['n_comments', 1],
    ['n_attachments', 1],
    ['n_votes', 1],
    ['n_watches', 1],
    ['n_issuelinks', 1],
    ['n_subtasks', 1],
    ['parent', 1],
    ['len_summary', 1],
    ['len_description', 1],
    ['len_comments', 1],
    ['avg_comment', 1],
    ['num_fix_versions', 1],
    ['num_affected_versions', 1]]


def feature_shuffling(model, X, Y, args):
    """
    Run the feature shuffling benchmark. This shuffles each feature
    individually and tests the ML performance after shuffling. If the
    performance of the ML model drops, the features is probably
    important, if the performance increases, the feature is probably
    noise.
    """
    splits = np.array_split(np.random.permutation(len(X)), 5)
    values = [0] * (len(property_lengths) + 1)
    labels = ['no shuffling']
    labels.extend([item[0] for item in property_lengths])

    for indices in splits:
        X_train = np.array([x for idx, x in enumerate(X) if idx not in indices])
        Y_train = np.array([y for idx, y in enumerate(Y) if idx not in indices])
        X_test = X[indices]
        Y_test = Y[indices]

        # Test performance ML without shuffling
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        values[0] += f1_score(Y_test, Y_pred, average='macro') / 5

        if args.confusion_matrix:
            print(f1_score(Y_test, Y_pred, average='macro'))
            if args.output_type == 'detection':
                cm_labels = ['Non-Architectural', 'Architectural']
            else:
                cm_labels = ['Executive', 'Property', 'Existence', 'Non-Architectural']
            cm = confusion_matrix(Y_test, Y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
            disp.plot()
            plt.show()

        start = 0
        for value_idx, property_length in enumerate(property_lengths):
            end = start + property_length[1]

            # shuffle single features
            permutation = np.random.permutation(len(X_train))
            X_shuffled = copy.copy(X_train)
            for idx, perm_idx in enumerate(permutation):
                X_shuffled[idx, start:end] = X_train[perm_idx, start:end]

            # Test performance again
            model.fit(X_shuffled, Y_train)
            Y_pred = model.predict(X_test)
            values[value_idx + 1] += f1_score(Y_test, Y_pred, average='macro') / 5

            start = end

    for idx in range(1, len(values)):
        values[idx] -= values[0]
    values[0] = 0

    fig, ax = plt.subplots()
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values)), labels=labels)
    ax.grid(linestyle='--')
    plt.show()

    return
