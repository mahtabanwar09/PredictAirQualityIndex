"""
app.py
------
Flask backend for the AQI Prediction web app.

Routes:
    /            Home page
    /predict     GET: show the input form. POST: run prediction, show result.
    /about       Project information page

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from utils import AQI_CATEGORIES, FEATURE_COLUMNS, get_aqi_category
from visualization_data import (
    CATEGORY_COLORS,
    build_visualization_dashboard,
    load_bulletin_data,
    load_station_data,
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "aqi_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "aqi_scaler.pkl"

# --- Load the regression model + scaler (required) ---------------------
ml_model = None
ml_scaler = None
ml_load_error = None
if MODEL_PATH.exists() and SCALER_PATH.exists():
    try:
        ml_model = joblib.load(MODEL_PATH)
        ml_scaler = joblib.load(SCALER_PATH)
    except Exception as exc:
        ml_load_error = f"Could not load trained model files: {exc}"
else:
    ml_load_error = "No trained model found. Run train_model.py first."

def run_prediction(input_values):
    X = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

    if ml_model is None or ml_scaler is None:
        raise RuntimeError(ml_load_error or "No trained model is available.")

    X_scaled = ml_scaler.transform(X)
    prediction = float(ml_model.predict(X_scaled)[0])
    model_used = "Machine Learning Regression"

    prediction = max(0, round(prediction))
    category, advice = get_aqi_category(prediction)
    return {
        "aqi": prediction,
        "category": category,
        "advice": advice,
        "model_used": model_used,
        "inputs": dict(zip(FEATURE_COLUMNS, input_values)),
    }


@app.route("/")
def home():
    return render_template(
        "index.html", aqi_categories=AQI_CATEGORIES, category_colors=CATEGORY_COLORS
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/visualizations")
def visualizations():
    dashboard = build_visualization_dashboard(request.args.get("city"))
    return render_template("visualizations.html", dashboard=dashboard)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # --- POST: read form inputs ---
    try:
        input_values = [float(request.form[col]) for col in FEATURE_COLUMNS]
    except (KeyError, ValueError):
        return render_template(
            "predict.html",
            error="Please fill in all fields with valid numbers.",
        )

    try:
        result = run_prediction(input_values)
    except RuntimeError as exc:
        return render_template(
            "predict.html",
            error=str(exc),
        )

    return render_template(
        "result.html",
        aqi=result["aqi"],
        category=result["category"],
        advice=result["advice"],
        model_used=result["model_used"],
        inputs=result["inputs"],
    )


def _streamlit_runtime():
    try:
        import streamlit as st
    except ImportError:
        return None

    return st if st.runtime.exists() else None


def _render_streamlit_app(st):
    st.set_page_config(page_title="AQI Prediction", layout="wide")
    _render_streamlit_styles(st)

    with st.sidebar:
        st.title("AQI")
        page = st.radio(
            "View",
            ["Predict", "Visualizations", "About"],
            label_visibility="collapsed",
        )

    if page == "Predict":
        _render_streamlit_predict(st)
    elif page == "Visualizations":
        _render_streamlit_visualizations(st)
    else:
        _render_streamlit_about(st)


def _render_streamlit_styles(st):
    st.markdown(
        """
        <style>
          .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
          }
          .aqi-hero {
            background: linear-gradient(135deg, #10233c, #1d4e89);
            border-radius: 8px;
            color: white;
            margin-bottom: 1.25rem;
            padding: 1.4rem 1.6rem;
          }
          .aqi-hero h1 {
            font-size: clamp(2rem, 4vw, 3.2rem);
            letter-spacing: 0;
            margin: 0 0 .35rem;
          }
          .aqi-hero p {
            color: rgba(255,255,255,.82);
            margin: 0;
          }
          .result-box {
            border: 1px solid #dbe7ef;
            border-radius: 8px;
            padding: 1.25rem;
          }
          .result-box strong {
            display: block;
            font-size: 3rem;
            line-height: 1;
            margin: .25rem 0 .65rem;
          }
          .category-pill {
            border-radius: 999px;
            color: white;
            display: inline-block;
            font-weight: 700;
            padding: .35rem .8rem;
          }
          .small-muted {
            color: #64748b;
            font-size: .92rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_streamlit_predict(st):
    st.markdown(
        """
        <div class="aqi-hero">
          <h1>Air Quality Index Prediction</h1>
          <p>Estimate AQI from pollutant and weather readings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    defaults = {
        "PM2.5": 85.0,
        "PM10": 140.0,
        "NO2": 30.0,
        "SO2": 12.0,
        "CO": 1.2,
        "O3": 45.0,
        "NH3": 20.0,
        "Temperature": 28.0,
        "Humidity": 55.0,
        "WindSpeed": 5.0,
    }
    labels = {
        "PM2.5": "PM2.5 (ug/m3)",
        "PM10": "PM10 (ug/m3)",
        "NO2": "NO2 (ug/m3)",
        "SO2": "SO2 (ug/m3)",
        "CO": "CO (mg/m3)",
        "O3": "O3 (ug/m3)",
        "NH3": "NH3 (ug/m3)",
        "Temperature": "Temperature (C)",
        "Humidity": "Humidity (%)",
        "WindSpeed": "Wind Speed (km/h)",
    }

    with st.form("aqi_prediction_form"):
        columns = st.columns(3)
        input_values = []
        for index, column in enumerate(FEATURE_COLUMNS):
            minimum = None if column == "Temperature" else 0.0
            maximum = 100.0 if column == "Humidity" else None
            value = columns[index % 3].number_input(
                labels[column],
                min_value=minimum,
                max_value=maximum,
                value=defaults[column],
                step=1.0,
            )
            input_values.append(float(value))

        submitted = st.form_submit_button("Predict AQI", type="primary")

    if not submitted:
        return

    try:
        result = run_prediction(input_values)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    color = CATEGORY_COLORS.get(result["category"], CATEGORY_COLORS["Unknown"])
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f"""
            <div class="result-box">
              <span class="small-muted">Predicted AQI</span>
              <strong>{result["aqi"]}</strong>
              <span class="category-pill" style="background:{color};">
                {result["category"]}
              </span>
              <p class="small-muted" style="margin-top:.9rem;">{result["model_used"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Health Guidance")
        st.write(result["advice"])
        st.dataframe(
            pd.DataFrame([result["inputs"]]).T.rename(columns={0: "Value"}),
            use_container_width=True,
        )


def _render_streamlit_visualizations(st):
    dashboard = build_visualization_dashboard()
    st.title("Air Quality Visualizations")

    if not dashboard["has_data"]:
        st.info("No extra dataset files were found.")
        return

    selected_city = st.selectbox(
        "City",
        dashboard["cities"],
        index=dashboard["cities"].index(dashboard["selected_city"]),
    )
    dashboard = build_visualization_dashboard(selected_city)

    metric_columns = st.columns(4)
    for column, card in zip(metric_columns, dashboard["summary_cards"]):
        column.metric(card["label"], card["value"], card["detail"])

    st.subheader(f"{dashboard['selected_city']} Monthly AQI Trend")
    bulletins = load_bulletin_data()
    city_frame = bulletins[bulletins["City"] == dashboard["selected_city"]].copy()
    monthly = (
        city_frame.set_index("date")["Index Value"]
        .sort_index()
        .resample("MS")
        .mean()
        .dropna()
        .tail(24)
    )
    if monthly.empty:
        st.info("No trend data available for this city.")
    else:
        st.line_chart(monthly.rename("AQI"), use_container_width=True)

    st.subheader("City Snapshot")
    selected_columns = st.columns(4)
    for column, card in zip(selected_columns, dashboard["selected_cards"]):
        column.metric(card["label"], card["value"], card["detail"])

    left, right = st.columns(2)
    with left:
        st.subheader("AQI Category Share")
        category_frame = pd.DataFrame(dashboard["category_items"])
        if not category_frame.empty:
            st.bar_chart(
                category_frame.set_index("name")["pct"].rename("Share %"),
                use_container_width=True,
            )

    with right:
        st.subheader("Highest AQI Cities")
        top_city_frame = pd.DataFrame(dashboard["top_cities"])
        if not top_city_frame.empty:
            top_city_frame["Average AQI"] = pd.to_numeric(top_city_frame["avg"])
            st.bar_chart(
                top_city_frame.set_index("name")["Average AQI"],
                use_container_width=True,
            )

    left, right = st.columns(2)
    with left:
        st.subheader("Bulletin Drivers")
        pollutant_frame = pd.DataFrame(dashboard["pollutant_items"])
        if not pollutant_frame.empty:
            pollutant_frame["Count"] = (
                pollutant_frame["count"].str.replace(",", "").astype(int)
            )
            st.bar_chart(
                pollutant_frame.set_index("name")["Count"],
                use_container_width=True,
            )

    with right:
        st.subheader("Station Map")
        stations = load_station_data().dropna(subset=["latitude", "longitude", "AQI"])
        if stations.empty:
            st.info("No station map data available.")
        else:
            map_frame = stations[["latitude", "longitude", "AQI"]].rename(
                columns={"latitude": "lat", "longitude": "lon"}
            )
            st.map(map_frame, use_container_width=True)


def _render_streamlit_about(st):
    st.title("About")
    st.write(
        "This app predicts AQI from pollutant and weather readings and classifies "
        "the result into the standard health categories."
    )

    rows = []
    lower_bound = 0
    for upper_bound, category, advice in AQI_CATEGORIES:
        rows.append(
            {
                "AQI Range": f"{lower_bound}-{upper_bound}",
                "Category": category,
                "Health Guidance": advice,
            }
        )
        lower_bound = upper_bound + 1

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


streamlit = _streamlit_runtime()
if streamlit is not None:
    _render_streamlit_app(streamlit)
elif __name__ == "__main__":
    app.run(debug=True)
