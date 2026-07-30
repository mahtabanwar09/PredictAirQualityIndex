"""
utils.py
--------
Shared helpers used by train_model.py, ann_model.py, and app.py.
Keeping this logic in one place avoids the category thresholds / feature
list drifting out of sync between training and serving.
"""

FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "NH3",
    "Temperature",
    "Humidity",
    "WindSpeed",
]

TARGET_COLUMN = "AQI"

# (upper_bound, category, health_advice)
AQI_CATEGORIES = [
    (50, "Good", "Air quality is satisfactory. Enjoy your usual outdoor activities."),
    (100, "Satisfactory", "Minor breathing discomfort possible for sensitive people."),
    (200, "Moderate", "Sensitive people (asthma, heart/lung conditions, children, elderly) should reduce prolonged outdoor exertion."),
    (300, "Poor", "Everyone may begin to experience breathing discomfort on prolonged exposure. Limit outdoor activity."),
    (400, "Very Poor", "Health warnings of emergency conditions. Avoid outdoor activity, especially for sensitive groups."),
    (500, "Severe", "Serious health effects for everyone. Stay indoors and use an air purifier if possible."),
]


def build_ann(input_dim: int, compile_model: bool = False):
    """
    Builds the Keras Sequential ANN model.

    Moved here to be shared between training and inference, making loading
    more robust by saving/loading weights only instead of the full model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        raise ImportError(
            "TensorFlow is required for ANN functionality. Please run 'pip install tensorflow'."
        )

    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),  # regression output
        ]
    )
    if compile_model:
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def get_aqi_category(aqi_value: float):
    """Return (category, health_advice) for a given AQI value."""
    # Clamp the value to a non-negative number
    aqi_value = max(0, aqi_value)
    for upper_bound, category, advice in AQI_CATEGORIES:
        if aqi_value <= upper_bound:
            return category, advice
    return AQI_CATEGORIES[-1][1], AQI_CATEGORIES[-1][2]  # For values > 500


def clean_dataframe(df):
    """Basic cleaning shared by training scripts: dedupe + drop missing target rows."""
    df = df.drop_duplicates().copy()
    # Impute missing feature values with column median (robust to outliers)
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=[TARGET_COLUMN])
    return df
