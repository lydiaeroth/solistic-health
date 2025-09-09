from flask import Flask, render_template, jsonify, request
import sqlite3
import pandas as pd
from datetime import datetime, timedelta


# ------- Fixed date. Make sure to update when uploading new data --------
END_DATE = datetime(2025, 9, 8)

app = Flask(__name__)
DB_Path = "data/dummy_health_data.db"

# ---------------- Utility function for fetching and resampling ----------------
def fetch_and_resample(data_type, start_date, freq, agg="mean", smoothing=1):
    conn = sqlite3.connect(DB_Path)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT startDate, value
        FROM health_data
        WHERE type=?
          AND datetime(substr(startDate,1,19)) >= ?
    """, (data_type, start_date.isoformat()))
    rows = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=["startDate", "value"])
    if df.empty:
        # Create an empty datetime index based on start_date and freq
        # Default to one point to avoid errors
        dt_index = pd.date_range(start=start_date, periods=1, freq=freq)
        return pd.DataFrame({"value": [0]}, index=dt_index)

    df["startDate"] = pd.to_datetime(df["startDate"].str[:19])
    df.set_index("startDate", inplace=True)

    if agg == "sum":
        df = df.resample(freq).sum()
    else:
        df = df.resample(freq).mean()
    
    df = df.rolling(smoothing, min_periods=1).mean()
    return df


# ---------------- Chart 1: Glucose & Steps by Hour ----------------
@app.route("/chart1")
def chart1():
    range_param = request.args.get("range", "24h")

    if range_param == "24h":
        start_date = END_DATE - timedelta(hours=24)
        freq, smoothing = "H", 1
    elif range_param == "3d":
        start_date = END_DATE - timedelta(days=3)
        freq, smoothing = "H", 3
    elif range_param == "2w":
        start_date = END_DATE - timedelta(weeks=2)
        freq, smoothing = "D", 1

    df_glucose = fetch_and_resample("HKQuantityTypeIdentifierBloodGlucose", start_date, freq, "mean", smoothing)
    df_steps = fetch_and_resample("HKQuantityTypeIdentifierStepCount", start_date, freq, "sum", smoothing)

    # Ensure DatetimeIndex and handle empty data
    for df in [df_glucose, df_steps]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.empty:
            df.index = pd.date_range(start=start_date, periods=1, freq=freq)
            df['value'] = [0]

    labels = df_glucose.index.strftime("%Y-%m-%d %H:%M").tolist()
    return jsonify({
        "labels": labels,
        "glucose": df_glucose["value"].tolist(),
        "steps": df_steps["value"].tolist()
    })


# ---------------- Chart 2: Heart Rate & Glucose by Month ----------------
@app.route("/chart2")
def chart2():
    # Look-back 12 months from END_DATE
    start_date = END_DATE - timedelta(days=365)
    
    # Resample both datasets monthly
    df_glucose = fetch_and_resample(
        "HKQuantityTypeIdentifierBloodGlucose",
        start_date,
        "MS",   # Month Start frequency
        "mean",
        1
    )
    df_hr = fetch_and_resample(
        "HKQuantityTypeIdentifierHeartRate",
        start_date,
        "MS",
        "mean",
        1
    )

    # Ensure datetime index
    for df in [df_glucose, df_hr]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # Create a full 12-month range
    all_months = pd.date_range(start=start_date, periods=12, freq="MS")

    # Reindex each dataset separately, filling missing months with 0
    df_glucose = df_glucose.reindex(all_months, fill_value=0)
    df_hr = df_hr.reindex(all_months, fill_value=0)

    # Create labels
    labels = all_months.strftime("%Y-%m").tolist()

    return jsonify({
        "labels": labels,
        "glucose": df_glucose["value"].tolist(),
        "heart_rate": df_hr["value"].tolist()
    })


# ---------------- Chart 3: Walking+Running Distance & Glucose ----------------
@app.route("/chart3")
def chart3():
    range_param = request.args.get("range", "14d")

    if range_param == "14d":
        start_date, freq, smoothing = END_DATE - timedelta(days=14), "D", 1
    elif range_param == "1m":
        start_date, freq, smoothing = END_DATE - timedelta(days=30), "D", 3
    elif range_param == "3m":
        start_date, freq, smoothing = END_DATE - timedelta(days=90), "D", 7
    elif range_param == "6m":
        start_date, freq, smoothing = END_DATE - timedelta(days=180), "D", 14

    df_glucose = fetch_and_resample("HKQuantityTypeIdentifierBloodGlucose", start_date, freq, "mean", smoothing)
    df_distance = fetch_and_resample("HKQuantityTypeIdentifierDistanceWalkingRunning", start_date, freq, "sum", smoothing)

    for df in [df_glucose, df_distance]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.empty:
            df.index = pd.date_range(start=start_date, periods=1, freq=freq)
            df['value'] = [0]

    labels = df_glucose.index.strftime("%Y-%m-%d").tolist()
    return jsonify({
        "labels": labels,
        "glucose": df_glucose["value"].tolist(),
        "distance": df_distance["value"].tolist()
    })


# ---------------- Chart 4: Calories Burned & Glucose ----------------
@app.route("/chart4")
def chart4():
    range_param = request.args.get("range", "14d")

    if range_param == "14d":
        start_date, freq, smoothing = END_DATE - timedelta(days=14), "D", 1
    elif range_param == "1m":
        start_date, freq, smoothing = END_DATE - timedelta(days=30), "D", 3
    elif range_param == "3m":
        start_date, freq, smoothing = END_DATE - timedelta(days=90), "D", 7
    elif range_param == "6m":
        start_date, freq, smoothing = END_DATE - timedelta(days=180), "D", 14

    df_glucose = fetch_and_resample("HKQuantityTypeIdentifierBloodGlucose", start_date, freq, "mean", smoothing)
    df_calories = fetch_and_resample("HKQuantityTypeIdentifierActiveEnergyBurned", start_date, freq, "sum", smoothing)

    for df in [df_glucose, df_calories]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.empty:
            df.index = pd.date_range(start=start_date, periods=1, freq=freq)
            df['value'] = [0]

    labels = df_glucose.index.strftime("%Y-%m-%d").tolist()
    return jsonify({
        "labels": labels,
        "glucose": df_glucose["value"].tolist(),
        "calories": df_calories["value"].tolist()
    })


# ---------------- Index ----------------
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
