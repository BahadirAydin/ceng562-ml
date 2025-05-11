from typing import Dict

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def build_models(random_state: int = 61) -> Dict[str, object]:
    """
    Return a dict mapping algorithm names to *unfitted* model instances.
    """
    return {
        "IsolationForest": IsolationForest(
            n_estimators=300,
            contamination=0.37,
            random_state=random_state,
            n_jobs=-1,
        ),
        # Setting `novelty=True` lets us call .predict
        # on test data after training.
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.37,
            novelty=True,
        ),
        "OneClassSVM": OneClassSVM(
            kernel="rbf",
            gamma="scale",
        ),
    }
