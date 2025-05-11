from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_wisconsin() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer()
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="label")
    return x, y
