import pandas as pd
import numpy as np
from pgmpy.estimators import BicScore
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMClassifier


def extract_features(data, p):
    approx = DecisionTreeRegressor()
    X = pd.get_dummies(data, columns = data.columns, prefix = "=")
    approx.fit(X, p)
    columns = list(data.columns)
    importance = np.zeros(data.shape[1])
    for i, feature in enumerate(X.columns):
        name = feature[0:feature.index("=")] if "=" in feature else feature
        importance[columns.index(name)] += approx.feature_importances_[i]
    return np.array(columns)[importance > 1e-8]

class ParentSetEstimator:
    def __init__(self, frame, initial_edges, tolerance = 0.01, max_estimators = 250):
        self.frame = frame
        self.initial_edges = initial_edges
        self.tolerance = tolerance
        self.max_estimators = max_estimators

    def estimate(self, variable, candidates):
        X = self.frame[candidates]
        y = self.frame[variable]

        y_unique = np.unique(y)
        if len(y_unique) < 2:
            # not enough data to filter candidates out
            return candidates
        
        y = LabelEncoder().fit_transform(y)
        model = LGBMClassifier(n_estimators = self.max_estimators, 
                               colsample_bytree = min(1, 50/len(candidates)))
        encoded_rows = OrdinalEncoder().fit_transform(X)
        data = pd.DataFrame(encoded_rows, columns = candidates, dtype = "category")
        model.fit(data, y)

        initial_parents = set([p for (p, n) in self.initial_edges if n == variable])
        for t_max in range(50, self.max_estimators, 50):
            p = model.predict_proba(data, start_iteration=0, num_iteration=t_max)
            parents = extract_features(data, p)
            if (initial_parents - set(parents))/len(initial_parents) <= self.tolerance:
                break 
        return parents

        

