"""
Apple Health XML Importer
=========================
Parses an Apple Health export.xml file and imports health data into a SQLite
database. Uses iterative XML parsing (iterparse) to handle large files
(250+ MB, 700K+ records) without loading the entire tree into memory.

Imports three categories of data into separate tables:
  - records:            Individual health measurements (glucose, steps, heart rate, etc.)
  - activity_summaries: Daily summaries from Apple Watch (exercise minutes, energy, stand hours)
  - workouts:           Workout sessions (cycling, running, walking, etc.) with statistics

Only 7 specific HKQuantityType identifiers are imported from <Record> elements
to keep the database lean. All other record types are discarded at parse time.

Usage:
  # As a module (called from app.py during upload):
  from import_health import import_from_xml
  count = import_from_xml("/path/to/export.xml", "/path/to/health_data.db")

  # Standalone (for testing against a local export.xml):
  python import_health.py
"""

import sqlite3
import xml.etree.ElementTree as ET
import os

# ---------------------------------------------------------------------------
# Only these Record types are imported. Everything else is filtered out.
# ---------------------------------------------------------------------------
ALLOWED_RECORD_TYPES = {
    "HKQuantityTypeIdentifierBloodGlucose",
    "HKQuantityTypeIdentifierStepCount",
    "HKQuantityTypeIdentifierHeartRate",
    "HKQuantityTypeIdentifierDistanceWalkingRunning",
    "HKQuantityTypeIdentifierActiveEnergyBurned",
    "HKQuantityTypeIdentifierBasalEnergyBurned",       # "Resting Energy"
    "HKQuantityTypeIdentifierDistanceCycling",
}

# How many rows to accumulate before flushing to SQLite (keeps memory low)
BATCH_SIZE = 5000


def parse_date(date_str):
    """
    Convert Apple Health date format to a SQLite-friendly format by stripping
    the timezone offset. All data is assumed to be in the user's local time.

    Input:  '2025-08-16 00:31:43 -0700'
    Output: '2025-08-16 00:31:43'
    """
    if date_str and len(date_str) > 19:
        return date_str[:19]
    return date_str


def _create_tables(cur):
    """
    Drop existing tables and create the three-table schema from scratch.
    This implements the replace-all strategy: every import starts fresh.
    """
    # Drop old tables
    cur.execute("DROP TABLE IF EXISTS records")
    cur.execute("DROP TABLE IF EXISTS activity_summaries")
    cur.execute("DROP TABLE IF EXISTS workouts")

    # records: individual health measurements
    cur.execute("""
        CREATE TABLE records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            source_name TEXT,
            unit TEXT,
            start_date TEXT NOT NULL,
            end_date TEXT,
            value REAL NOT NULL
        )
    """)

    # activity_summaries: daily summaries (exercise minutes, energy rings)
    cur.execute("""
        CREATE TABLE activity_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            apple_exercise_time REAL,
            active_energy_burned REAL,
            active_energy_burned_goal REAL,
            apple_stand_hours INTEGER,
            apple_stand_hours_goal INTEGER
        )
    """)

    # workouts: exercise sessions with distance/energy statistics
    cur.execute("""
        CREATE TABLE workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_type TEXT NOT NULL,
            duration REAL,
            source_name TEXT,
            start_date TEXT NOT NULL,
            end_date TEXT,
            total_distance REAL,
            total_distance_unit TEXT,
            total_energy_burned REAL,
            total_energy_burned_unit TEXT
        )
    """)


def _create_indexes(cur):
    """
    Create indexes after all data is inserted. Building indexes after bulk
    inserts is faster than maintaining them during inserts.
    """
    cur.execute("CREATE INDEX idx_records_type_date ON records (type, start_date)")
    cur.execute("CREATE INDEX idx_records_date ON records (start_date)")
    cur.execute("CREATE INDEX idx_activity_date ON activity_summaries (date)")
    cur.execute("CREATE INDEX idx_workouts_type_date ON workouts (activity_type, start_date)")


def _flush_records(cur, batch):
    """Insert a batch of record rows into the records table."""
    if not batch:
        return
    cur.executemany("""
        INSERT INTO records (type, source_name, unit, start_date, end_date, value)
        VALUES (?, ?, ?, ?, ?, ?)
    """, batch)


def _flush_activities(cur, batch):
    """Insert a batch of activity summary rows. Uses INSERT OR REPLACE
    to handle duplicate dates (shouldn't happen, but defensive)."""
    if not batch:
        return
    cur.executemany("""
        INSERT OR REPLACE INTO activity_summaries
        (date, apple_exercise_time, active_energy_burned,
         active_energy_burned_goal, apple_stand_hours, apple_stand_hours_goal)
        VALUES (?, ?, ?, ?, ?, ?)
    """, batch)


def _flush_workouts(cur, batch):
    """Insert a batch of workout rows into the workouts table."""
    if not batch:
        return
    cur.executemany("""
        INSERT INTO workouts
        (activity_type, duration, source_name, start_date, end_date,
         total_distance, total_distance_unit, total_energy_burned, total_energy_burned_unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)


def import_from_xml(xml_path, db_path):
    """
    Parse export.xml and import records, activity summaries, and workouts
    into the SQLite database. Uses replace-all strategy: drops existing
    tables before importing.

    This is the main entry point called by app.py during file upload.

    Args:
        xml_path: Absolute path to the export.xml file
        db_path:  Absolute path to the SQLite database file

    Returns:
        Total number of rows imported across all three tables
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Use WAL mode for better concurrent read/write performance
    cur.execute("PRAGMA journal_mode=WAL")

    # Drop and recreate all tables (replace-all strategy)
    _create_tables(cur)
    conn.commit()

    # Accumulators for batch inserts
    record_batch = []
    activity_batch = []
    workout_batch = []
    total_count = 0

    # Track the current workout being parsed (Workout elements have children)
    workout_pending = None

    # Stream-parse the XML file to keep memory usage constant (~50 MB)
    # regardless of file size. Each element is cleared after processing.
    context = ET.iterparse(xml_path, events=("start", "end"))

    for event, elem in context:

        # ----- Record elements (individual health measurements) -----
        if event == "end" and elem.tag == "Record":
            rec_type = elem.attrib.get("type", "")
            if rec_type in ALLOWED_RECORD_TYPES:
                value = elem.attrib.get("value")
                if value is not None:
                    try:
                        record_batch.append((
                            rec_type,
                            elem.attrib.get("sourceName"),
                            elem.attrib.get("unit"),
                            parse_date(elem.attrib.get("startDate")),
                            parse_date(elem.attrib.get("endDate")),
                            float(value),
                        ))
                    except (ValueError, TypeError):
                        pass  # skip records with non-numeric values

            # Flush batch if full
            if len(record_batch) >= BATCH_SIZE:
                _flush_records(cur, record_batch)
                total_count += len(record_batch)
                record_batch = []

            # Free memory — clear the element and remove references
            elem.clear()

        # ----- ActivitySummary elements (daily rings/exercise data) -----
        elif event == "end" and elem.tag == "ActivitySummary":
            date_comp = elem.attrib.get("dateComponents")
            if date_comp:
                activity_batch.append((
                    date_comp,
                    _safe_float(elem.attrib.get("appleExerciseTime")),
                    _safe_float(elem.attrib.get("activeEnergyBurned")),
                    _safe_float(elem.attrib.get("activeEnergyBurnedGoal")),
                    _safe_int(elem.attrib.get("appleStandHours")),
                    _safe_int(elem.attrib.get("appleStandHoursGoal")),
                ))
            elem.clear()

        # ----- Workout elements (start tag — begin accumulating) -----
        elif event == "start" and elem.tag == "Workout":
            workout_pending = {
                "activity_type": elem.attrib.get("workoutActivityType", ""),
                "duration": _safe_float(elem.attrib.get("duration")),
                "source_name": elem.attrib.get("sourceName"),
                "start_date": parse_date(elem.attrib.get("startDate")),
                "end_date": parse_date(elem.attrib.get("endDate")),
                "total_distance": None,
                "total_distance_unit": None,
                "total_energy_burned": None,
                "total_energy_burned_unit": None,
            }

        # ----- WorkoutStatistics (children of Workout) -----
        elif event == "end" and elem.tag == "WorkoutStatistics" and workout_pending:
            stat_type = elem.attrib.get("type", "")
            # Capture distance statistics (cycling, walking/running, etc.)
            if "Distance" in stat_type:
                workout_pending["total_distance"] = _safe_float(elem.attrib.get("sum"))
                workout_pending["total_distance_unit"] = elem.attrib.get("unit")
            # Capture energy burned statistics
            elif "EnergyBurned" in stat_type:
                workout_pending["total_energy_burned"] = _safe_float(elem.attrib.get("sum"))
                workout_pending["total_energy_burned_unit"] = elem.attrib.get("unit")

        # ----- Workout end tag — finalize and add to batch -----
        elif event == "end" and elem.tag == "Workout":
            if workout_pending:
                workout_batch.append((
                    workout_pending["activity_type"],
                    workout_pending["duration"],
                    workout_pending["source_name"],
                    workout_pending["start_date"],
                    workout_pending["end_date"],
                    workout_pending["total_distance"],
                    workout_pending["total_distance_unit"],
                    workout_pending["total_energy_burned"],
                    workout_pending["total_energy_burned_unit"],
                ))
                workout_pending = None
            elem.clear()

    # Flush any remaining rows in the batches
    _flush_records(cur, record_batch)
    total_count += len(record_batch)

    _flush_activities(cur, activity_batch)
    total_count += len(activity_batch)

    _flush_workouts(cur, workout_batch)
    total_count += len(workout_batch)

    # Build indexes after all inserts (faster than maintaining during inserts)
    _create_indexes(cur)

    conn.commit()
    conn.close()

    return total_count


def _safe_float(value):
    """Convert a string to float, returning None if the value is missing or invalid."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value):
    """Convert a string to int, returning None if the value is missing or invalid."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Standalone mode: run directly to import a local export.xml for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "export.xml")
    db_path = os.path.join(script_dir, "health_data.db")

    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        print("Place your Apple Health export.xml in the same directory as this script.")
    else:
        print(f"Importing from {xml_path} ...")
        count = import_from_xml(xml_path, db_path)
        print(f"Done. Imported {count:,} total rows into {db_path}")
