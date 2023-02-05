from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import pandas as pd


def select_features(rfc, data, target):
    selector = SelectFromModel(rfc, threshold=0.05)
    imp_features = selector.fit_transform(data, target)
    return imp_features


def run_randomforests(data, target):
    rfc = RandomForestClassifier(n_estimators=100,
                                 max_depth=20,
                                 bootstrap=True,
                                 criterion="gini",
                                 max_features=None,
                                 n_jobs=-1)
    rfc.fit(data, target)
    print(rfc.score(data, target))
    imp_features = select_features(rfc, data, target)
    #print(pd.DataFrame(imp_features))

