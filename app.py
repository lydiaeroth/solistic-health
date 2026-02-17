"""
Static Health Dashboard — Flask Backend
========================================
A health data visualization dashboard for Type 1 diabetes management.
Displays Apple Health data with blood glucose as the primary reference
metric, overlaid with activity, cardio, cycling, and movement data.

Architecture:
  - Data is uploaded via the dashboard UI as an Apple Health export.zip
  - The zip is parsed by import_health.py into a SQLite database
  - This Flask app serves a single-page ECharts dashboard and JSON APIs
  - All charts share a global time range controlled by the frontend

Routes:
  GET  /              — Serve the dashboard (index.html)
  GET  /api/data      — All chart data for the selected time range
  GET  /api/snapshot  — Today's Snapshot card values (glucose, steps, exercise)
  POST /api/upload    — Upload & import Apple Health export.zip (password-protected)

Deployment: Render (paid tier with persistent disk for SQLite)
"""

import os
import tempfile
import zipfile

from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

from import_health import import_from_xml

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Max upload size: 300 MB (Apple Health exports are typically 150-280 MB)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

# Database path: reads from env var (for Render persistent disk) or defaults to local
DB_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "health_data.db"),
)

# Upload password: set via environment variable on Render
UPLOAD_PASSWORD = os.environ.get("UPLOAD_PASSWORD", "changeme")

# ---------------------------------------------------------------------------
# Time range configuration
# Maps the range parameter from the frontend to lookback duration,
# pandas resampling frequency, and smoothing window size.
# ---------------------------------------------------------------------------
RANGE_CONFIG = {
    "24h": {"delta": timedelta(hours=24), "freq": "h",  "smoothing": 1},
    "7d":  {"delta": timedelta(days=7),   "freq": "h",  "smoothing": 3},
    "14d": {"delta": timedelta(days=14),  "freq": "D",  "smoothing": 1},
    "30d": {"delta": timedelta(days=30),  "freq": "D",  "smoothing": 3},
    "3mo": {"delta": timedelta(days=90),  "freq": "D",  "smoothing": 7},
    "6mo": {"delta": timedelta(days=180), "freq": "D",  "smoothing": 14},
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_latest_data_date():
    """
    Return the most recent timestamp in the records table as a datetime.
    Used as the dynamic "end date" for all time range calculations,
    replacing the old hardcoded END_DATE.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT MAX(start_date) FROM records")
    row = cur.fetchone()
    conn.close()
    if row and row[0]:
        dt = datetime.strptime(row[0][:19], "%Y-%m-%d %H:%M:%S")
        # Truncate to the start of the hour so the common time axis aligns
        # with pandas resample boundaries (which truncate to period starts)
        return dt.replace(minute=0, second=0, microsecond=0)
    return datetime.now().replace(minute=0, second=0, microsecond=0)


def db_has_data():
    """
    Check if the records table exists and has any data.
    Used to show an empty state on the dashboard before first upload.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("SELECT COUNT(*) FROM records")
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def fetch_and_resample(data_type, start_date, end_date, freq, agg="mean", smoothing=1):
    """
    Query the records table for a specific health data type within a date
    range, resample to the given frequency, and apply rolling smoothing.

    This is the core data processing function. Every chart metric passes
    through this function.

    Args:
        data_type:  HKQuantityTypeIdentifier string (e.g. "...BloodGlucose")
        start_date: datetime, inclusive start of the range
        end_date:   datetime, inclusive end of the range
        freq:       pandas frequency string ("h" for hourly, "D" for daily)
        agg:        "mean" for averaged metrics (glucose, heart rate)
                    "sum" for cumulative metrics (steps, calories, distance)
        smoothing:  rolling window size (1 = no smoothing)

    Returns:
        pandas Series with DatetimeIndex. Missing periods are NaN (not 0),
        which is critical for showing data gaps on charts.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT start_date, value
        FROM records
        WHERE type = ?
          AND start_date >= ?
          AND start_date <= ?
        ORDER BY start_date
    """
    df = pd.read_sql_query(
        query, conn,
        params=(data_type, start_date.strftime("%Y-%m-%d %H:%M:%S"),
                end_date.strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.close()

    # If no data exists for this metric in this range, return empty Series
    if df.empty:
        idx = pd.date_range(start=start_date, end=end_date, freq=freq)
        return pd.Series(dtype=float, index=idx, name="value")

    df["start_date"] = pd.to_datetime(df["start_date"])
    df.set_index("start_date", inplace=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Resample to the target frequency
    # min_count=1 on sum() ensures empty groups return NaN (not 0) for gap visibility
    if agg == "sum":
        resampled = df["value"].resample(freq).sum(min_count=1)
    else:
        resampled = df["value"].resample(freq).mean()

    # Apply rolling smoothing (smoothing=1 means no smoothing)
    if smoothing > 1:
        resampled = resampled.rolling(smoothing, min_periods=1).mean()

    return resampled


def fetch_exercise_minutes(start_date, end_date, freq):
    """
    Query the activity_summaries table for daily exercise minutes.
    This data comes from <ActivitySummary> elements in the Apple Health
    export (requires Apple Watch).

    The data is inherently daily, so if freq is hourly, each hour within
    a day gets the full day's exercise minutes (not divided by 24).
    """
    conn = sqlite3.connect(DB_PATH)

    # Check if table exists (it may not if no ActivitySummary data was imported)
    try:
        df = pd.read_sql_query(
            """
            SELECT date, apple_exercise_time
            FROM activity_summaries
            WHERE date >= ? AND date <= ?
            ORDER BY date
            """,
            conn,
            params=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        idx = pd.date_range(start=start_date, end=end_date, freq=freq)
        return pd.Series(dtype=float, index=idx, name="value")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Resample — already daily data, so daily freq is a no-op
    # For hourly freq, forward-fill to show daily value at each hour
    if freq == "h":
        result = df["apple_exercise_time"].resample("D").sum(min_count=1)
        result = result.resample(freq).ffill()
    else:
        result = df["apple_exercise_time"].resample(freq).sum(min_count=1)

    return result


def series_to_list(series, common_idx):
    """
    Align a pandas Series to a common DatetimeIndex and convert to a list
    suitable for JSON serialization. NaN values become None (JSON null),
    which tells ECharts to break the line and show a visible gap.
    """
    aligned = series.reindex(common_idx)
    return [None if pd.isna(v) else round(float(v), 2) for v in aligned]


def extract_export_xml(zip_path, extract_to):
    """
    Extract export.xml from an Apple Health export.zip file.
    Handles both common zip layouts:
      - export.xml at the root of the zip
      - apple_health_export/export.xml in a subdirectory
    Returns the path to the extracted XML, or None if not found.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        for candidate in ["export.xml", "apple_health_export/export.xml"]:
            if candidate in names:
                z.extract(candidate, extract_to)
                return os.path.join(extract_to, candidate)
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page dashboard."""
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    """
    Return all chart data for the requested time range in a single JSON
    response. The frontend calls this once on page load and again whenever
    the user changes the global time range picker. All 10 charts are
    populated from this single payload (no waterfall of requests).

    Query params:
        range: one of 24h, 7d, 14d, 30d, 3mo, 6mo (default: 7d)

    Response: JSON with labels array and one array per metric. Missing
    data points are null (not 0) so charts show visible gaps.
    """
    # Return empty data if no health data has been imported yet
    if not db_has_data():
        return jsonify({
            "labels": [],
            "glucose": [], "steps": [], "heart_rate": [],
            "active_calories": [], "resting_energy": [],
            "walking_running_distance": [], "cycling_distance": [],
            "exercise_minutes": [],
            "avg_glucose": None, "avg_daily_steps": None,
            "avg_exercise_minutes": None,
            "range": "7d", "freq": "h",
        })

    range_param = request.args.get("range", "7d")
    config = RANGE_CONFIG.get(range_param, RANGE_CONFIG["7d"])

    end_date = get_latest_data_date()
    start_date = end_date - config["delta"]
    freq = config["freq"]
    smoothing = config["smoothing"]

    # Normalize dates to align with pandas resample boundaries:
    # - Hourly freq: truncate to start of hour (already done in get_latest_data_date)
    # - Daily freq: truncate to start of day (midnight) so resampled dates match
    if freq == "D":
        start_date = start_date.replace(hour=0, minute=0, second=0)
        end_date = end_date.replace(hour=23, minute=59, second=59)

    # Fetch all metrics — each returns a pandas Series with DatetimeIndex
    glucose = fetch_and_resample(
        "HKQuantityTypeIdentifierBloodGlucose",
        start_date, end_date, freq, "mean", smoothing
    )
    steps = fetch_and_resample(
        "HKQuantityTypeIdentifierStepCount",
        start_date, end_date, freq, "sum", smoothing
    )
    heart_rate = fetch_and_resample(
        "HKQuantityTypeIdentifierHeartRate",
        start_date, end_date, freq, "mean", smoothing
    )
    active_cal = fetch_and_resample(
        "HKQuantityTypeIdentifierActiveEnergyBurned",
        start_date, end_date, freq, "sum", smoothing
    )
    resting_energy = fetch_and_resample(
        "HKQuantityTypeIdentifierBasalEnergyBurned",
        start_date, end_date, freq, "sum", smoothing
    )
    walk_dist = fetch_and_resample(
        "HKQuantityTypeIdentifierDistanceWalkingRunning",
        start_date, end_date, freq, "sum", smoothing
    )
    cycle_dist = fetch_and_resample(
        "HKQuantityTypeIdentifierDistanceCycling",
        start_date, end_date, freq, "sum", smoothing
    )
    exercise_min = fetch_exercise_minutes(start_date, end_date, freq)

    # Align all series to a common time axis
    common_idx = pd.date_range(start=start_date, end=end_date, freq=freq)
    fmt = "%Y-%m-%d %H:%M" if freq == "h" else "%Y-%m-%d"

    # ---- Snapshot averages for the selected time range ----
    conn = sqlite3.connect(DB_PATH)
    ds = start_date.strftime("%Y-%m-%d %H:%M:%S")
    de = end_date.strftime("%Y-%m-%d %H:%M:%S")

    # Average glucose reading
    cur = conn.execute("""
        SELECT AVG(value) FROM records
        WHERE type = 'HKQuantityTypeIdentifierBloodGlucose'
          AND start_date >= ? AND start_date <= ?
    """, (ds, de))
    row = cur.fetchone()
    avg_glucose = round(row[0], 1) if row and row[0] else None

    # Average daily steps (total steps / days in range)
    cur = conn.execute("""
        SELECT SUM(value) FROM records
        WHERE type = 'HKQuantityTypeIdentifierStepCount'
          AND start_date >= ? AND start_date <= ?
    """, (ds, de))
    row = cur.fetchone()
    total_steps_sum = row[0] if row and row[0] else None
    num_days = max(1, config["delta"].days)
    avg_daily_steps = round(total_steps_sum / num_days) if total_steps_sum else None

    # Average daily exercise minutes
    try:
        cur = conn.execute("""
            SELECT AVG(apple_exercise_time) FROM activity_summaries
            WHERE date >= ? AND date <= ?
        """, (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        row = cur.fetchone()
        avg_exercise = round(row[0], 1) if row and row[0] else None
    except Exception:
        avg_exercise = None

    conn.close()

    return jsonify({
        "labels": common_idx.strftime(fmt).tolist(),
        "glucose": series_to_list(glucose, common_idx),
        "steps": series_to_list(steps, common_idx),
        "heart_rate": series_to_list(heart_rate, common_idx),
        "active_calories": series_to_list(active_cal, common_idx),
        "resting_energy": series_to_list(resting_energy, common_idx),
        "walking_running_distance": series_to_list(walk_dist, common_idx),
        "cycling_distance": series_to_list(cycle_dist, common_idx),
        "exercise_minutes": series_to_list(exercise_min, common_idx),
        "avg_glucose": avg_glucose,
        "avg_daily_steps": avg_daily_steps,
        "avg_exercise_minutes": avg_exercise,
        "range": range_param,
        "freq": freq,
    })


@app.route("/api/snapshot")
def api_snapshot():
    """
    Return today's snapshot values for the header cards:
      - latest_glucose:   Most recent glucose reading (mg/dL)
      - total_steps:      Cumulative step count for the latest data day
      - exercise_minutes: Exercise minutes for the latest data day

    Uses the latest data date (not wall-clock today) so the snapshot
    always shows meaningful values even if data isn't real-time.
    """
    if not db_has_data():
        return jsonify({
            "latest_glucose": None,
            "total_steps": None,
            "exercise_minutes": None,
            "as_of": None,
        })

    conn = sqlite3.connect(DB_PATH)
    latest_date = get_latest_data_date()
    day_start = latest_date.strftime("%Y-%m-%d")

    # Latest glucose: the single most recent reading
    cur = conn.execute("""
        SELECT value FROM records
        WHERE type = 'HKQuantityTypeIdentifierBloodGlucose'
        ORDER BY start_date DESC LIMIT 1
    """)
    row = cur.fetchone()
    latest_glucose = round(row[0]) if row else None

    # Total steps on the latest data day
    cur = conn.execute("""
        SELECT SUM(value) FROM records
        WHERE type = 'HKQuantityTypeIdentifierStepCount'
          AND start_date >= ?
    """, (day_start,))
    row = cur.fetchone()
    total_steps = int(row[0]) if row and row[0] else None

    # Exercise minutes from activity_summaries (may not exist)
    try:
        cur = conn.execute("""
            SELECT apple_exercise_time FROM activity_summaries
            WHERE date = ?
        """, (day_start,))
        row = cur.fetchone()
        exercise_min = int(row[0]) if row and row[0] else None
    except Exception:
        exercise_min = None

    conn.close()

    return jsonify({
        "latest_glucose": latest_glucose,
        "total_steps": total_steps,
        "exercise_minutes": exercise_min,
        "as_of": day_start,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Handle Apple Health export.zip upload. This is the only authenticated
    endpoint — it requires a password that matches the UPLOAD_PASSWORD
    environment variable.

    Flow:
      1. Validate password from the form data
      2. Save the uploaded zip to a temp directory
      3. Extract export.xml from the zip
      4. Call import_from_xml() which drops all tables and re-imports
      5. Return the number of records imported

    The import can take 30-90 seconds for large files (250+ MB).
    Gunicorn must be configured with --timeout 300 to avoid killing
    the request mid-parse.
    """
    # Authenticate
    password = request.form.get("password", "")
    if password != UPLOAD_PASSWORD:
        return jsonify({"error": "Invalid password"}), 401

    # Validate file presence and type
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.endswith(".zip"):
        return jsonify({"error": "File must be a .zip"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded zip to temp directory
            zip_path = os.path.join(tmpdir, secure_filename(file.filename))
            file.save(zip_path)

            # Extract export.xml from the zip
            xml_path = extract_export_xml(zip_path, tmpdir)
            if not xml_path:
                return jsonify({"error": "export.xml not found in zip"}), 400

            # Import data (drops existing tables, re-imports everything)
            count = import_from_xml(xml_path, DB_PATH)

        return jsonify({"status": "success", "records_imported": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
