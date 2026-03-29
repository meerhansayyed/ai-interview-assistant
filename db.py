import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "interview_data.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        # Interviews table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                username            TEXT    NOT NULL,
                domain              TEXT    NOT NULL,
                created_at          TEXT    NOT NULL,
                overall_score       INTEGER,
                hire_recommendation TEXT,
                scorecard_json      TEXT,
                sentiment_log_json  TEXT,
                transcript_json     TEXT
            )
        """)
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username              TEXT PRIMARY KEY,
                email                 TEXT UNIQUE,
                password              TEXT NOT NULL,
                role                  TEXT DEFAULT 'user',
                created_at            TEXT NOT NULL,
                interviews_completed  INTEGER DEFAULT 0
            )
        """)
        conn.commit()


# ==========================
# USER FUNCTIONS
# ==========================

def create_user(username, email, password, role="user"):
    with get_conn() as conn:
        try:
            conn.execute("""
                INSERT INTO users (username, email, password, role, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                username.strip().lower(),
                email.strip().lower(),
                password,
                role,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def authenticate_user(username, password):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT username, role FROM users WHERE username = ? AND password = ?",
            (username.strip().lower(), password)
        ).fetchone()
        return dict(row) if row else None


def get_user_stats(username):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT interviews_completed FROM users WHERE username = ?",
            (username.strip().lower(),)
        ).fetchone()
        return dict(row) if row else {}


def get_all_users():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT username, email, role, created_at, interviews_completed FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_user(username):
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username.strip().lower(),))
        conn.commit()


# ==========================
# INTERVIEW FUNCTIONS
# ==========================

def save_interview(username, domain, overall_score, hire_recommendation,
                   scorecard, sentiment_log, messages):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO interviews
              (username, domain, created_at, overall_score, hire_recommendation,
               scorecard_json, sentiment_log_json, transcript_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            domain,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_score,
            hire_recommendation,
            json.dumps(scorecard),
            json.dumps(sentiment_log),
            json.dumps(messages),
        ))
        # Increment user's interview count
        conn.execute("""
            UPDATE users SET interviews_completed = interviews_completed + 1
            WHERE username = ?
        """, (username,))
        conn.commit()


def get_all_interviews():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, username, domain, created_at,
                   overall_score, hire_recommendation
            FROM interviews
            ORDER BY created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_interview_by_id(interview_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM interviews WHERE id = ?", (interview_id,)
        ).fetchone()
    if row:
        d = dict(row)
        d["scorecard"]     = json.loads(d.pop("scorecard_json",     "{}"))
        d["sentiment_log"] = json.loads(d.pop("sentiment_log_json", "[]"))
        d["transcript"]    = json.loads(d.pop("transcript_json",    "[]"))
        return d
    return None


def get_interviews_by_user(username: str):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, username, domain, created_at,
                   overall_score, hire_recommendation
            FROM interviews
            WHERE username = ?
            ORDER BY created_at DESC
        """, (username,)).fetchall()
    return [dict(r) for r in rows]


def delete_interview(interview_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
        conn.commit()


def get_stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM interviews").fetchone()[0]
        avg_score = conn.execute(
            "SELECT AVG(overall_score) FROM interviews"
        ).fetchone()[0] or 0

        hire_counts = conn.execute("""
            SELECT hire_recommendation, COUNT(*) as cnt
            FROM interviews
            GROUP BY hire_recommendation
            ORDER BY cnt DESC
        """).fetchall()

        domain_counts = conn.execute("""
            SELECT domain, COUNT(*) as cnt
            FROM interviews
            GROUP BY domain
            ORDER BY cnt DESC
        """).fetchall()

        recent = conn.execute("""
            SELECT username, domain, overall_score, hire_recommendation, created_at
            FROM interviews
            ORDER BY created_at DESC
            LIMIT 5
        """).fetchall()

    return {
        "total":         total,
        "avg_score":     round(float(avg_score), 1),
        "hire_counts":   [dict(r) for r in hire_counts],
        "domain_counts": [dict(r) for r in domain_counts],
        "recent":        [dict(r) for r in recent],
    }