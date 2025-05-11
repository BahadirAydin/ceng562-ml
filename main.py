import logging
import pandas as pd

from data_loader import load_wisconsin
from evaluate import evaluate, plot_curves, plot_metric_bars
from models import build_models
from preprocess import split_and_scale

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)


def main():
    logging.info("Loading data …")
    x, y = load_wisconsin()

    logging.info("Splitting and scaling …")
    x_train, x_test, y_train, y_test, scaler = split_and_scale(x, y)

    logging.info("Creating models …")
    models = build_models()

    results = []

    x_train_benign = x_train[y_train == 1]
    for name, model in models.items():
        logging.info(f"Training ➜ {name}")
        model.fit(x_train_benign)

        logging.info(f"Evaluating ➜ {name}")
        metrics = evaluate(model, x_test, y_test, anomaly_label=0)
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
