"""Evaluate all three stages on chronological future-month folds."""
from pathlib import Path
import numpy as np
import pandas as pd
from freight.features import weight_medians_from, prepare_data, add_features
from freight.model import FreightModel

FOLDS = [
    ("Jan-Jun -> Jul-Aug", 6, 7, 8),
    ("Jan-Aug -> Sep-Oct", 8, 9, 10),
    ("Jan-Sep -> Oct", 9, 10, 10),
]


def scores(actual, predicted):
    error = np.abs(actual-predicted)
    return {
        "MAE": error.mean(),
        "RMSE": np.sqrt(np.mean((actual-predicted)**2)),
        "MAPE (%)": np.mean(error/actual)*100,
    }


def run_validation():
    data = pd.read_csv("data/train_test.csv", parse_dates=["date"])
    results = []
    for label, train_end, val_start, val_end in FOLDS:
        train_raw = data[data["date"].dt.month <= train_end]
        val_raw = data[data["date"].dt.month.between(val_start, val_end)]

        # Recalculate statistics for this fold, never using validation targets.
        medians = weight_medians_from(train_raw)
        train = add_features(prepare_data(train_raw, medians))
        val = add_features(prepare_data(val_raw, medians))

        # One complete fit also provides stage 1 and stage 2 predictions.
        model = FreightModel(stage=3).fit(train)
        for stage, name in [(1, "Robust regression"),
                            (2, "+ LightGBM"),
                            (3, "+ Lane correction")]:
            predicted = model.predict(val, stage=stage)
            results.append({"Fold": label, "Model": name,
                            "Rows": len(val), **scores(val["posted_rate"].to_numpy(), predicted)})
            print(f"{label:23} {name:19} MAE ${results[-1]['MAE']:.2f}", flush=True)

    results = pd.DataFrame(results)
    Path("outputs").mkdir(exist_ok=True)
    results.to_csv("outputs/cv_results.csv", index=False)
    print("\n", results.round(2).to_string(index=False))
    return results


if __name__ == "__main__":
    run_validation()
