import logging
import pandas as pd

from data_loader import load_wisconsin
from evaluate import evaluate, plot_curves, plot_metric_bars
from models import build_models
from preprocess import split_and_scale
from tuning import tune_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)

hyperparameters = {
    "IsolationForest": {
        "n_estimators": [50, 100, 200, 300, 500],
        "contamination": [0.05, 0.1, 0.15, 0.2, 0.3, 0.37, 0.4],
    },
    "LocalOutlierFactor": {
        "n_neighbors": [2, 5, 10, 15, 20, 25],
        "contamination": [0.05, 0.1, 0.15, 0.2, 0.3, 0.37, 0.4],
    },
    "OneClassSVM": {
        "kernel": ["rbf", "sigmoid"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "nu": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    },
}


def main():
    logging.info("Loading data …")
    x, y = load_wisconsin()

    logging.info("Splitting and scaling …")
    x_train, x_test, y_train, y_test, scaler = split_and_scale(x, y)

    logging.info("Creating models …")
    models = build_models()

    results = []

    x_train_benign = x_train[y_train == 1]
    for name, model in build_models().items():
        logging.info(f"Tuning ➜ {name}")
        param_grid = hyperparameters.get(name, {})
        tuned_model, best_params = tune_model(
            name, model, param_grid, x_train_benign, x_test, y_test, anomaly_label=0
        )

        logging.info(f"Best Params for {name}: {best_params}")
        logging.info(f"Evaluating ➜ {name}")
        metrics = evaluate(tuned_model, x_test, y_test, anomaly_label=0)
        plot_curves(name, metrics)
        metrics["model"] = name
        results.append(metrics)

    summary_df = (
        pd.DataFrame(results)
        .set_index("model")[["pr_auc", "roc_auc"]]
        .sort_values("pr_auc", ascending=False)
    )
    plot_metric_bars(summary_df)
    logging.info("Figures written to ./figures/")

    print("\n=== Outlier-detection summary (higher is better) ===")
    print(summary_df.to_markdown(floatfmt=".4f"))


if __name__ == "__main__":
    main()
