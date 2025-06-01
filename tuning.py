from itertools import product
from sklearn.base import clone
from evaluate import evaluate


def tune_model(name, base_model, param_grid, x_train, x_test, y_test, anomaly_label=0):
    best_model = None
    best_score = -float("inf")
    best_params = None

    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        model = clone(base_model).set_params(**params)
        model.fit(x_train)

        metrics = evaluate(model, x_test, y_test, anomaly_label=anomaly_label)
        score = metrics["pr_auc"]

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params
