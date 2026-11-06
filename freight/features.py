"""Clean and create the features actually used by the prediction model."""
import numpy as np
import pandas as pd

EQUIPMENT_CODES = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}


def weight_medians_from(train):
    """Compute replacement weights using training observations only."""
    weights = train.assign(weight=train["weight"].abs())
    return weights.groupby("equipment")["weight"].median()


def prepare_data(df, weight_medians, daily_market=None):
    """Apply the same cleaning to training, validation, and December inputs."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # The 292 negative training weights have plausible magnitudes.
    data["weight_missing"] = data["weight"].isna().astype(int)
    data["weight"] = data["weight"].abs()
    data["weight"] = data["weight"].fillna(data["equipment"].map(weight_medians))

    # Daily mean smooths the small row-to-row fluctuations in market_index.
    # For a prediction period, the supplied, unlabeled features provide this mean.
    if daily_market is None:
        daily_market = data.groupby("date")["market_index"].mean()
    data["market_index"] = data["date"].map(daily_market)
    return data


def add_features(data):
    """Return model features; all calculations work on future dates as well."""
    x = data.copy()
    x["log_distance"] = np.log(x["distance"])
    x["inverse_distance"] = 1 / x["distance"]
    x["log_weight"] = np.log(x["weight"])
    x["equipment_code"] = x["equipment"].map(EQUIPMENT_CODES)

    # Approximate straight-line distance from city coordinates (in miles).
    lat1, lon1, lat2, lon2 = [np.radians(x[c]) for c in
        ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]]
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    air_miles = 3958.8 * 2 * np.arcsin(np.sqrt(a))
    x["route_ratio"] = x["distance"] / np.maximum(air_miles, 1)

    day = x["date"].dt
    x["weekday"] = day.dayofweek
    months = (x["date"] - pd.Timestamp("2025-01-01")).dt.days / 30.4
    x["months_elapsed"] = months
    x["months_squared"] = months**2
    x["annual_sin"] = np.sin(2*np.pi*months/12)
    x["annual_cos"] = np.cos(2*np.pi*months/12)
    month_progress = (day.day-1) / (day.days_in_month-1)
    x["quarter_end_ramp"] = np.where(day.month.isin([3,6,9,12]), month_progress, 0)
    return x


# Only columns consumed by LightGBM appear here.
TREE_FEATURES = [
    "log_distance", "distance", "route_ratio", "log_weight", "weight_missing",
    "equipment_code", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "quote_signal"
]
