import os
import time
import sqlite3
import json
import logging
from datetime import datetime
from backend.config import MONGO_URI, DB_NAME, SQLITE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Database")

# Global clients
mongo_client = None
db_type = "sqlite"

try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError

    # Try connecting to MongoDB with a short timeout (1.5 seconds)
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
    # Trigger a call to check if connection works
    mongo_client.server_info()
    db_type = "mongodb"
    logger.info("Connected successfully to MongoDB.")
except Exception as e:
    logger.warning(f"MongoDB connection failed: {e}. Falling back to SQLite.")
    mongo_client = None
    db_type = "sqlite"

# Ensure SQLite is initialized if fallback
def init_sqlite():
    if db_type == "sqlite":
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speech_records (
                id TEXT PRIMARY KEY,
                patient_id TEXT,
                audio_path TEXT,
                timestamp TEXT,
                features TEXT,
                ssi_score REAL,
                prediction_score REAL,
                severity TEXT,
                status TEXT,
                logs TEXT
            )
        """)
        conn.commit()
        conn.close()

init_sqlite()

def save_speech_record(record):
    """
    Saves a speech processing record.
    record schema:
    {
        "id": str,
        "patient_id": str,
        "audio_path": str,
        "timestamp": str,
        "features": dict, (pitch_var, pause_raw, zcr, articulation_raw)
        "ssi_score": float,
        "prediction_score": float,
        "severity": str,
        "status": str,
        "logs": list of str
    }
    """
    # Standardize timestamp
    if "timestamp" not in record:
        record["timestamp"] = datetime.utcnow().isoformat()

    if db_type == "mongodb" and mongo_client:
        try:
            db = mongo_client[DB_NAME]
            db.records.update_one({"id": record["id"]}, {"$set": record}, upsert=True)
            logger.info(f"Saved record {record['id']} to MongoDB.")
            return True
        except Exception as e:
            logger.error(f"Failed to save to MongoDB: {e}. Attempting SQLite backup.")
    
    # SQLite Fallback
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO speech_records 
            (id, patient_id, audio_path, timestamp, features, ssi_score, prediction_score, severity, status, logs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["patient_id"],
                record["audio_path"],
                record["timestamp"],
                json.dumps(record.get("features", {})),
                record.get("ssi_score", 0.0),
                record.get("prediction_score", 0.0),
                record.get("severity", "Unknown"),
                record.get("status", "Completed"),
                json.dumps(record.get("logs", []))
            )
        )
        conn.commit()
        conn.close()
        logger.info(f"Saved record {record['id']} to SQLite fallback.")
        return True
    except Exception as e:
        logger.error(f"Failed to save to SQLite: {e}")
        return False

def get_speech_records():
    """Retrieves all speech processing records, sorted by timestamp descending."""
    if db_type == "mongodb" and mongo_client:
        try:
            db = mongo_client[DB_NAME]
            cursor = db.records.find({}, {"_id": 0}).sort("timestamp", -1)
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to fetch from MongoDB: {e}. Falling back to SQLite.")

    # SQLite fallback
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM speech_records ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "id": row["id"],
                "patient_id": row["patient_id"],
                "audio_path": row["audio_path"],
                "timestamp": row["timestamp"],
                "features": json.loads(row["features"]) if row["features"] else {},
                "ssi_score": row["ssi_score"],
                "prediction_score": row["prediction_score"],
                "severity": row["severity"],
                "status": row["status"],
                "logs": json.loads(row["logs"]) if row["logs"] else []
            })
        return records
    except Exception as e:
        logger.error(f"Failed to fetch from SQLite: {e}")
        return []

def get_patient_records(patient_id):
    """Retrieves all records for a specific patient."""
    if db_type == "mongodb" and mongo_client:
        try:
            db = mongo_client[DB_NAME]
            cursor = db.records.find({"patient_id": patient_id}, {"_id": 0}).sort("timestamp", -1)
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to fetch from MongoDB: {e}. Falling back to SQLite.")

    # SQLite fallback
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM speech_records WHERE patient_id = ? ORDER BY timestamp DESC", (patient_id,))
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "id": row["id"],
                "patient_id": row["patient_id"],
                "audio_path": row["audio_path"],
                "timestamp": row["timestamp"],
                "features": json.loads(row["features"]) if row["features"] else {},
                "ssi_score": row["ssi_score"],
                "prediction_score": row["prediction_score"],
                "severity": row["severity"],
                "status": row["status"],
                "logs": json.loads(row["logs"]) if row["logs"] else []
            })
        return records
    except Exception as e:
        logger.error(f"Failed to fetch from SQLite: {e}")
        return []

def get_db_info():
    """Returns database type and state details."""
    return {
        "db_type": db_type,
        "status": "connected" if (db_type == "mongodb" or os.path.exists(SQLITE_PATH)) else "error"
    }
