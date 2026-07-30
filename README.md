# Air Quality Index (AQI) Prediction System

A web app that predicts AQI from pollutant and weather readings using a
classical ML regression model, then shows the AQI category and a health
recommendation. An ANN model can also be trained locally with the optional
TensorFlow dependencies.

## Project Structure

```text
AQI_Project/
|-- dataset/
|   |-- generate_dataset.py
|   `-- air_quality.csv
|-- datasets/
|   `-- *_AQIBulletins.csv
|-- models/
|   |-- aqi_model.pkl
|   |-- aqi_scaler.pkl
|   |-- model_metrics.pkl
|   |-- aqi_ann.h5
|   `-- ann_scaler.pkl
|-- static/css/style.css
|-- templates/
|-- utils.py
|-- visualization_data.py
|-- train_model.py
|-- ann_model.py
|-- app.py
|-- requirements.txt
`-- requirements-training.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For local ANN training or XGBoost experiments, install the optional training
dependencies too:

```bash
pip install -r requirements-training.txt
```

## 1. Get A Dataset

A synthetic dataset is already generated at `dataset/air_quality.csv` so you
can run the whole pipeline immediately. To use real data, replace that file
with a dataset that has these exact column names:

```text
PM2.5, PM10, NO2, SO2, CO, O3, NH3, Temperature, Humidity, WindSpeed, AQI
```

You can also re-run `python dataset/generate_dataset.py` to regenerate the
synthetic dataset.

## 2. Train The Models

```bash
python train_model.py
python ann_model.py
```

Both scripts read `dataset/air_quality.csv` and write to `models/`. The app
works with just `train_model.py` output. The ANN is optional; if TensorFlow is
not installed or the ANN files cannot be loaded, the ANN option is disabled.

## 3. Run The App

Streamlit:

```bash
streamlit run app.py
```

Flask:

```bash
python app.py
```

Visit `http://127.0.0.1:5000` for the Flask version.

## Notes On The Included Synthetic Dataset

`dataset/generate_dataset.py` builds a realistic but synthetic dataset with
8,000 rows, missing values, and duplicate rows so cleaning, EDA, training, and
saving have real work to do out of the box. It is not real air-quality data;
for a final project, swap in an actual dataset with matching column names
before your final training run.

## Deployment

`app.py` supports both Streamlit Community Cloud and Flask hosting.

For Streamlit Community Cloud, use `app.py` as the entry point and install
`requirements.txt`. TensorFlow is intentionally excluded from the hosted
runtime dependencies because the ANN is optional and TensorFlow wheels may not
be available for the newest Python versions used by the platform.

For Render/Railway-style Flask hosting, the included `Procfile` still works:
`gunicorn app:app`.

Make sure `models/aqi_model.pkl` and `models/aqi_scaler.pkl` are trained and
present before deploying. The ANN files are optional; if TensorFlow is not
installed, the app disables the ANN option and keeps the regression model
available.
