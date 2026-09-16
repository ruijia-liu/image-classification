"""Nested group-aware selection and local metrics."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


def checked_folds(X, y, groups, n_splits, seed):
    if n_splits < 2:
        raise ValueError("At least two folds are required")
    for label in (0, 1):
        if len(np.unique(groups[y == label])) < n_splits:
            raise ValueError("Insufficient independent groups per class for requested folds")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups))
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Group leakage detected")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("Every training and evaluation fold must contain both classes")
    return folds


def metrics(y, prediction, groups):
    def scores(a, b):
        return {"accuracy": float(accuracy_score(a, b)),
                "balanced_accuracy": float(balanced_accuracy_score(a, b)),
                "f1": float(f1_score(a, b, zero_division=0))}
    unique = np.unique(groups)
    truth = [y[groups == g][0] for g in unique]
    vote = [int(np.mean(prediction[groups == g]) >= 0.5) for g in unique]
    return {"patch": scores(y, prediction), "group": scores(truth, vote)}


def model_grid(seed):
    return [
        {"clf": [SVC(kernel="linear", class_weight="balanced")], "clf__C": [0.1, 1, 10]},
        {"clf": [SVC(kernel="rbf", class_weight="balanced")],
         "clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.1]},
        {"clf": [LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)],
         "clf__C": [0.1, 1, 10]},
        {"clf": [RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)],
         "clf__min_samples_leaf": [1, 3]},
        {"clf": [ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=seed)],
         "clf__min_samples_leaf": [1, 3]},
    ]


def evaluate(X, y, groups, outer=3, inner=2, seed=42, grid=None):
    rows = []
    for fold, (train, test) in enumerate(checked_folds(X, y, groups, outer, seed), 1):
        inner_folds = checked_folds(X[train], y[train], groups[train], inner, seed + fold)
        pipeline = Pipeline([("scale", RobustScaler()), ("clf", SVC())])
        search = GridSearchCV(pipeline, model_grid(seed) if grid is None else grid,
                              cv=inner_folds, scoring="balanced_accuracy", error_score="raise", n_jobs=1)
        search.fit(X[train], y[train])
        prediction = search.predict(X[test])
        params = {k: v for k, v in search.best_params_.items() if k != "clf"}
        rows.append({"fold": fold, "model": type(search.best_estimator_["clf"]).__name__,
                     "parameters": params, **metrics(y[test], prediction, groups[test])})
    return {"method": "nested_group_cv", "seed": seed, "outer_folds": outer,
            "inner_folds": inner, "folds": rows}
