"""Train the final model and generate both required prediction CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
from freight.features import weight_medians_from, prepare_data, add_features
from freight.model import FreightModel


def coordinates(train, validation):
    """City coordinates for the December dataset, which has no coordinate columns."""
    places = []
    for data in [train, validation]:
        for side in ["pickup", "delivery"]:
            names = data[[side, side+"_lat", side+"_lon"]].copy()
            names.columns = ["city", "lat", "lon"]
            places.append(names)
    return pd.concat(places).drop_duplicates("city").set_index("city")


def main():
    train_raw = pd.read_csv("data/train_test.csv")
    val_raw = pd.read_csv("data/validation.csv")
    december_raw = pd.read_csv("data/december_chart_inputs.csv")
    template = pd.read_csv("data/validation_predictions_template.csv")
    medians = weight_medians_from(train_raw)

    train = add_features(prepare_data(train_raw, medians))
    val = add_features(prepare_data(val_raw, medians))
    model = FreightModel(stage=3).fit(train)
    print("Training observations excluded from robust fit:", model.outlier_count)

    # The final predictions must match the provided template IDs exactly.
    result = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val["load_id"], "predicted_rate": model.predict(val)}),
        on="load_id", how="left", validate="one_to_one"
    )
    result["predicted_rate"] = result["predicted_rate"].round(2)
    result.to_csv("validation_predictions.csv", index=False)

    # December provides no coordinates or market signals, so use the supplied
    # unlabeled November–December shipments to obtain same-day market features.
    city = coordinates(train_raw, val_raw)
    december = december_raw.drop(columns="predicted_rate").copy()
    december["date"] = pd.to_datetime(december["date"])
    for side in ["pickup", "delivery"]:
        december[side+"_lat"] = december[side].map(city["lat"])
        december[side+"_lon"] = december[side].map(city["lon"])

    val_dates = val_raw.assign(date=pd.to_datetime(val_raw["date"]))
    daily_market = val_dates.groupby("date")["market_index"].mean()
    daily_quote = val_dates.groupby("date")["quote_signal"].mean()
    december["quote_signal"] = december["date"].map(daily_quote)
    december = add_features(prepare_data(december, medians, daily_market))

    december_result = december_raw.copy()
    december_result["predicted_rate"] = model.predict(december).round(2)
    Path("outputs").mkdir(exist_ok=True)
    december_result.to_csv("outputs/december_chart_predictions.csv", index=False)
    print("Final predictions:",len(result),"December dates:",len(december_result))


if __name__ == "__main__":
    main()
