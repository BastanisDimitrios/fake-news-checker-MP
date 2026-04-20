from __future__ import annotations

from typing import Optional, Tuple, Dict
import os
import re
import hmac
import uuid
import string
import sqlite3
import hashlib
import secrets as pysecrets
import base64
import json
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import extra_streamlit_components as stx
import smtplib
from email.message import EmailMessage

from updater import update_reference_lists
from credibility_engine import evaluate_source


# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="Fake News Credibility Checker",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Paths
# ============================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = PROJECT_ROOT / "users.db"

MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# Limits / settings
# ============================================================
MAX_TEXT_CHARS = 50_000
RATE_LIMIT_SECONDS = 2
SESSION_DAYS = 30
RESET_CODE_TTL_MIN = 15
RESET_COOLDOWN_SEC = 45


# ============================================================
# Cookie manager
# ============================================================
def cookie_manager() -> stx.CookieManager:
    if "_cookie_manager" not in st.session_state:
        st.session_state["_cookie_manager"] = stx.CookieManager(key="cookie_manager_main")
    return st.session_state["_cookie_manager"]



# ============================================================
# Secrets / config
# ============================================================
def get_app_secret() -> str:
    secret = st.secrets.get("APP_SECRET", "")
    if not secret or len(str(secret)) < 20:
        raise RuntimeError("Missing or weak APP_SECRET in .streamlit/secrets.toml")
    return str(secret)
def make_remember_token(email: str, days_valid: int = SESSION_DAYS) -> str:
    expires = int((datetime.utcnow() + timedelta(days=days_valid)).timestamp())
    payload = {
        "email": email.lower().strip(),
        "exp": expires,
    }

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    secret = get_app_secret().encode("utf-8")

    sig = hmac.new(secret, payload_json.encode("utf-8"), hashlib.sha256).hexdigest()

    token_obj = {
        "payload": payload,
        "sig": sig,
    }

    token_json = json.dumps(token_obj, separators=(",", ":"))
    return base64.urlsafe_b64encode(token_json.encode("utf-8")).decode("utf-8")


def verify_remember_token(token: str) -> Optional[str]:
    try:
        token_json = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        token_obj = json.loads(token_json)

        payload = token_obj["payload"]
        sig = token_obj["sig"]

        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        secret = get_app_secret().encode("utf-8")

        expected_sig = hmac.new(
            secret,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig, expected_sig):
            return None

        exp = int(payload["exp"])
        if int(datetime.utcnow().timestamp()) > exp:
            return None

        email = str(payload["email"]).lower().strip()
        return email if get_user(email) else None

    except Exception:
        return None

def smtp_config() -> Dict[str, str]:
    return {
        "host": str(st.secrets.get("SMTP_HOST", "")).strip(),
        "port": str(st.secrets.get("SMTP_PORT", "587")).strip(),
        "user": str(st.secrets.get("SMTP_USER", "")).strip(),
        "pass": str(st.secrets.get("SMTP_PASS", "")).strip(),
        "from": str(st.secrets.get("SMTP_FROM", "")).strip(),
    }


def smtp_is_ready(cfg: Dict[str, str]) -> bool:
    return bool(cfg["host"] and cfg["user"] and cfg["pass"] and cfg["from"])


def rate_limit_gate(action_key: str, seconds: int = RATE_LIMIT_SECONDS) -> None:
    now = datetime.utcnow()
    k = f"last_{action_key}"
    last = st.session_state.get(k)

    if last and (now - last) < timedelta(seconds=seconds):
        st.warning("Please wait a moment before repeating the same action.")
        st.stop()

    st.session_state[k] = now


# ============================================================
# Theme
# ============================================================
def get_theme() -> str:
    if "theme" not in st.session_state:
        cm = cookie_manager()
        v = cm.get("theme")
        st.session_state.theme = v if v in ("dark", "light") else "light"
    return st.session_state.theme


def set_theme(theme: str) -> None:
    theme = "dark" if theme == "dark" else "light"
    st.session_state.theme = theme
    cookie_manager().set(
        "theme",
        theme,
        expires_at=datetime.utcnow() + timedelta(days=365),
    )


def inject_css(theme: str) -> None:
    if theme == "dark":
        bg = "#0b1220"
        panel = "#0f172a"
        panel2 = "#111c31"
        panel3 = "#162036"
        border = "rgba(148,163,184,0.18)"
        text = "#E5E7EB"
        muted = "#94A3B8"
        input_bg = "#0f172a"
        accent = "#60a5fa"
        accent2 = "rgba(96,165,250,0.14)"
        accent3 = "#22c55e"
        danger = "#ef4444"
        warn = "#f59e0b"
        shadow = "0 14px 35px rgba(0,0,0,0.35)"
        hero_grad = "linear-gradient(135deg, #0f172a 0%, #172554 50%, #1e3a8a 100%)"
    else:
        bg = "#f4f7fb"
        panel = "#ffffff"
        panel2 = "#f8fafc"
        panel3 = "#eef4fb"
        border = "rgba(15,23,42,0.10)"
        text = "#0F172A"
        muted = "#64748B"
        input_bg = "#ffffff"
        accent = "#2563eb"
        accent2 = "rgba(37,99,235,0.10)"
        accent3 = "#16a34a"
        danger = "#dc2626"
        warn = "#d97706"
        shadow = "0 14px 35px rgba(15,23,42,0.10)"
        hero_grad = "linear-gradient(135deg, #1e3a8a 0%, #2563eb 45%, #0f172a 100%)"

    st.markdown(
        f"""
<style>
:root {{
  --bg:{bg};
  --panel:{panel};
  --panel2:{panel2};
  --panel3:{panel3};
  --border:{border};
  --text:{text};
  --muted:{muted};
  --inputbg:{input_bg};
  --accent:{accent};
  --accent2:{accent2};
  --accent3:{accent3};
  --danger:{danger};
  --warn:{warn};
  --shadow:{shadow};
  --hero:{hero_grad};
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--bg) !important;
}}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
label, p, span, div, h1, h2, h3, h4, h5, h6,
li, small, strong, em {{
  color: var(--text) !important;
}}

.small, .stCaption, [data-testid="stCaptionContainer"] {{
  color: var(--muted) !important;
}}

section[data-testid="stSidebar"] {{
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}}

.block-container {{
  padding-top: 1.4rem !important;
  padding-bottom: 2rem !important;
}}

.card {{
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 22px;
  padding: 18px;
  box-shadow: var(--shadow);
}}

.card-lite {{
  background: var(--panel2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px;
  padding: 16px;
}}

.card-soft {{
  background: var(--panel3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px;
  padding: 16px;
}}

.hero {{
  background: var(--hero);
  border-radius: 26px;
  padding: 30px;
  color: white !important;
  box-shadow: var(--shadow);
  border: 1px solid rgba(255,255,255,0.08);
}}

.hero * {{
  color: white !important;
}}

.hero-title {{
  font-size: 32px;
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1.05;
}}

.hero-sub {{
  margin-top: 12px;
  font-size: 14px;
  line-height: 1.6;
  opacity: 0.94;
  max-width: 780px;
}}

.hero-kicker {{
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.14);
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 14px;
}}

.metric-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  box-shadow: var(--shadow);
}}

.metric-title {{
  font-size: 12px;
  color: var(--muted) !important;
  margin-bottom: 6px;
}}

.metric-value {{
  font-size: 24px;
  font-weight: 900;
  letter-spacing: -0.03em;
}}

.metric-good {{
  color: var(--accent3) !important;
}}

.metric-warn {{
  color: var(--warn) !important;
}}

.metric-bad {{
  color: var(--danger) !important;
}}

.badge {{
  display:inline-block;
  padding: 6px 11px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--panel);
  font-size: 12px;
  color: var(--text);
  margin-right: 8px;
  margin-bottom: 6px;
}}

.hero-badge {{
  display:inline-block;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 12px;
  color: white !important;
  margin-right: 8px;
  margin-bottom: 8px;
}}

.navwrap {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}}

.navtitle {{
  font-weight: 900;
  font-size: 18px;
  letter-spacing: -0.02em;
}}

.navsub {{
  margin-top: 2px;
  font-size: 12px;
  color: var(--muted) !important;
}}

.auth-switch {{
  background: var(--panel);
  padding: 8px 10px 2px 10px;
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
}}

.result-panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px;
  box-shadow: var(--shadow);
}}

.section-title {{
  font-size: 18px;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 8px;
}}

.section-sub {{
  font-size: 13px;
  color: var(--muted) !important;
  margin-bottom: 12px;
}}

.dashboard-shell {{
  display: flex;
  flex-direction: column;
  gap: 18px;
}}

.side-panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 18px;
  box-shadow: var(--shadow);
  height: 100%;
}}

.side-panel-title {{
  font-size: 18px;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 4px;
}}

.side-panel-sub {{
  font-size: 13px;
  color: var(--muted) !important;
  margin-bottom: 14px;
}}

.soft-divider {{
  height: 1px;
  background: var(--border);
  margin: 14px 0;
}}

.stat-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}}

.stat-card-pro {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px;
  box-shadow: var(--shadow);
}}

.stat-label {{
  font-size: 12px;
  color: var(--muted) !important;
  margin-bottom: 8px;
}}

.stat-value {{
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -0.03em;
}}

.stat-note {{
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted) !important;
}}

.workspace-grid {{
  display: grid;
  grid-template-columns: 1.45fr 0.85fr;
  gap: 18px;
}}

.prob-track {{
  width: 100%;
  height: 12px;
  border-radius: 999px;
  overflow: hidden;
  background: var(--panel3);
  border: 1px solid var(--border);
}}

.prob-real {{
  height: 100%;
  background: linear-gradient(90deg, var(--accent3), var(--accent));
}}

.prob-fake {{
  height: 100%;
  background: linear-gradient(90deg, var(--danger), var(--warn));
}}

.big-label {{
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -0.03em;
}}

.muted-mini {{
  font-size: 12px;
  color: var(--muted) !important;
}}

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {{
  background: var(--inputbg) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {{
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}}

input::placeholder, textarea::placeholder {{
  color: var(--muted) !important;
  opacity: 0.9;
}}

div[data-testid="stRadio"] > div {{
  gap: 8px !important;
}}

div[data-testid="stRadio"] label {{
  background: var(--panel2) !important;
  border: 1px solid var(--border) !important;
  padding: 8px 14px !important;
  border-radius: 999px !important;
}}

div[data-testid="stTabs"] button {{
  border-radius: 999px !important;
}}

div[data-testid="stTabs"] button[aria-selected="true"] {{
  background: var(--accent2) !important;
  border: 1px solid var(--border) !important;
}}

div[data-testid="stToggle"] {{
  transform: scale(1.04);
}}

div[data-testid="stToggle"] label p {{
  color: var(--text) !important;
}}

div[data-testid="stToggle"] input:checked + div {{
  background: var(--accent) !important;
}}

div.stButton > button {{
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  font-weight: 700 !important;
  min-height: 44px !important;
  box-shadow: var(--shadow);
}}

div.stDownloadButton > button {{
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  font-weight: 700 !important;
  min-height: 44px !important;
}}

[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
}}

@media (max-width: 900px) {{
  .stat-grid {{
    grid-template-columns: 1fr;
  }}
  .workspace-grid {{
    grid-template-columns: 1fr;
  }}
}}

.about-hero {{
  background: var(--hero);
  border-radius: 24px;
  padding: 28px;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: var(--shadow);
  color: white !important;
}}

.about-hero * {{
  color: white !important;
}}

.about-hero-title {{
  font-size: 30px;
  font-weight: 900;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin-bottom: 10px;
}}

.about-hero-sub {{
  font-size: 14px;
  line-height: 1.7;
  opacity: 0.95;
  max-width: 860px;
}}

.about-pill {{
  display: inline-block;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 12px;
  margin-right: 8px;
  margin-top: 10px;
}}

.feature-grid-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}

.feature-grid-3 {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}}

.about-feature-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px;
  box-shadow: var(--shadow);
}}

.about-feature-icon {{
  font-size: 24px;
  margin-bottom: 10px;
}}

.about-feature-title {{
  font-size: 17px;
  font-weight: 800;
  margin-bottom: 8px;
  letter-spacing: -0.02em;
}}

.about-feature-text {{
  font-size: 13px;
  line-height: 1.7;
  color: var(--muted) !important;
}}

.about-highlight {{
  background: linear-gradient(135deg, var(--accent2), transparent);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px;
  box-shadow: var(--shadow);
}}

.about-list {{
  margin: 0;
  padding-left: 18px;
}}

.about-list li {{
  margin-bottom: 8px;
  color: var(--muted) !important;
}}

.tech-chip {{
  display: inline-block;
  padding: 8px 12px;
  border-radius: 12px;
  background: var(--panel2);
  border: 1px solid var(--border);
  font-size: 12px;
  margin-right: 8px;
  margin-bottom: 8px;
}}

@media (max-width: 900px) {{
  .feature-grid-2,
  .feature-grid-3 {{
    grid-template-columns: 1fr;
  }}
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DB helpers
# ============================================================
def db_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _has_column(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols


def init_db() -> None:
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            display_name TEXT,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            input_type TEXT NOT NULL,
            source TEXT,
            prediction TEXT,
            score REAL,
            p_fake REAL,
            p_real REAL,
            threshold REAL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    if not _has_column(cur, "users", "display_name"):
        cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
    if not _has_column(cur, "users", "last_login_at"):
        cur.execute("ALTER TABLE users ADD COLUMN last_login_at TEXT")
    if not _has_column(cur, "history", "threshold"):
        cur.execute("ALTER TABLE history ADD COLUMN threshold REAL")

    conn.commit()
    conn.close()


# ============================================================
# Password hashing
# ============================================================
def pbkdf2_hash_password(password: str, pepper: str) -> str:
    password = password.strip()
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    salt = os.urandom(16)
    iterations = 210_000
    data = (password + pepper).encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", data, salt, iterations, dklen=32)
    return f"{iterations}${salt.hex()}${dk.hex()}"


def pbkdf2_verify_password(password: str, stored: str, pepper: str) -> bool:
    try:
        it_s, salt_hex, dk_hex = stored.split("$")
        iterations = int(it_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)

        data = (password.strip() + pepper).encode("utf-8")
        dk = hashlib.pbkdf2_hmac("sha256", data, salt, iterations, dklen=32)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ============================================================
# Users
# ============================================================
def register_user(email: str, password: str, display_name: Optional[str] = None) -> None:
    email = email.lower().strip()
    pw_hash = pbkdf2_hash_password(password, get_app_secret())

    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, pw_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
        (email, pw_hash, (display_name or "").strip() or None, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user(email: str) -> Optional[Tuple[str, str, Optional[str]]]:
    email = email.lower().strip()
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT email, pw_hash, display_name FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return row


def update_last_login(email: str) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET last_login_at=? WHERE email=?",
        (datetime.utcnow().isoformat(), email),
    )
    conn.commit()
    conn.close()


def update_display_name(email: str, display_name: str) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET display_name=? WHERE email=?",
        ((display_name or "").strip() or None, email),
    )
    conn.commit()
    conn.close()


def update_password(email: str, new_password: str) -> None:
    new_hash = pbkdf2_hash_password(new_password, get_app_secret())
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET pw_hash=? WHERE email=?", (new_hash, email))
    conn.commit()
    conn.close()


# ============================================================
# Sessions / auto-login
# ============================================================
def create_session(email: str, days_valid: int = SESSION_DAYS) -> None:
    email = email.lower().strip()
    expires = datetime.utcnow() + timedelta(days=days_valid)
    remember_token = make_remember_token(email, days_valid=days_valid)

    st.session_state["_set_remember_token"] = remember_token
    st.session_state["_set_remember_expires"] = expires
    st.session_state["user_email"] = email


def apply_pending_cookie_writes() -> None:
    cm = cookie_manager()

    if "_set_remember_token" in st.session_state:
        cm.set(
            "remember_token",
            st.session_state["_set_remember_token"],
            expires_at=st.session_state["_set_remember_expires"],
        )
        del st.session_state["_set_remember_token"]

    if "_set_remember_expires" in st.session_state:
        del st.session_state["_set_remember_expires"]

    if st.session_state.get("_clear_remember_cookies"):
        expired = datetime.utcnow() - timedelta(days=1)
        cm.set("remember_token", "", expires_at=expired)
        st.session_state["_clear_remember_cookies"] = False


def delete_session() -> None:
    st.session_state["_clear_remember_cookies"] = True
    st.session_state["user_email"] = None


def restore_session_from_cookie() -> None:
    if "user_email" not in st.session_state:
        st.session_state.user_email = None

    if st.session_state.user_email:
        return

    token = cookie_manager().get("remember_token")

    if not token:
        return

    email = verify_remember_token(token)

    if email:
        st.session_state.user_email = email
    else:
        st.session_state["_clear_remember_cookies"] = True
        st.session_state.user_email = None


# ============================================================
# History
# ============================================================
def save_history(
    email: str,
    input_type: str,
    source: str,
    pred: str,
    score: float,
    p_fake: float,
    p_real: float,
    threshold: float,
) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO history (email, input_type, source, prediction, score, p_fake, p_real, threshold, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            email,
            input_type,
            source,
            pred,
            score,
            p_fake,
            p_real,
            float(threshold),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def load_history(email: str, limit: int = 500) -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query(
        """
        SELECT created_at, input_type, source, prediction, score, p_fake, p_real, threshold
        FROM history
        WHERE email=?
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(email, limit),
    )
    conn.close()
    return df


def clear_history(email: str) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE email=?", (email,))
    conn.commit()
    conn.close()


def get_user_stats(email: str) -> Dict[str, float]:
    df = load_history(email, limit=5000)

    if df.empty:
        return {
            "total": 0,
            "text_count": 0,
            "url_count": 0,
            "csv_count": 0,
            "avg_score": 0.0,
        }

    return {
        "total": int(len(df)),
        "text_count": int((df["input_type"] == "TEXT").sum()),
        "url_count": int((df["input_type"] == "URL").sum()),
        "csv_count": int((df["input_type"] == "CSV").sum()),
        "avg_score": float(df["score"].mean()),
    }


def load_recent_history(email: str, limit: int = 5) -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query(
        """
        SELECT created_at, input_type, source, prediction, score
        FROM history
        WHERE email=?
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(email, limit),
    )
    conn.close()
    return df


# ============================================================
# Password reset
# ============================================================
def _gen_reset_code() -> str:
    return f"{pysecrets.randbelow(1_000_000):06d}"


def create_reset_code(email: str) -> str:
    code = _gen_reset_code()
    expires = datetime.utcnow() + timedelta(minutes=RESET_CODE_TTL_MIN)

    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO password_resets (email, code, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (email.lower().strip(), code, expires.isoformat(), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return code


def verify_reset_code(email: str, code: str) -> bool:
    email = email.lower().strip()
    code = (code or "").strip()

    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT code, expires_at FROM password_resets WHERE email=? ORDER BY id DESC LIMIT 1",
        (email,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return False

    db_code, exp = row
    try:
        if datetime.utcnow() > datetime.fromisoformat(exp):
            return False
    except Exception:
        return False

    return hmac.compare_digest(str(db_code), str(code))


def send_reset_email(to_email: str, code: str) -> None:
    cfg = smtp_config()

    if not smtp_is_ready(cfg):
        raise RuntimeError("SMTP is not configured in secrets.toml")

    msg = EmailMessage()
    msg["Subject"] = "Password Reset Verification Code"
    msg["From"] = cfg["from"]
    msg["To"] = to_email
    msg.set_content(
        f"""
Fake News Credibility Checker

Your password reset verification code is:

{code}

This code will expire in {RESET_CODE_TTL_MIN} minutes.

If you did not request this, please ignore this email.

Best regards,
Fake News Credibility System
"""
    )

    host = cfg["host"]
    port = int(cfg["port"]) if cfg["port"].isdigit() else 587
    user = cfg["user"]
    pw = cfg["pass"]

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.starttls()
        server.login(user, pw)
        server.send_message(msg)


# ============================================================
# Text cleaning / model
# ============================================================
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(model, vectorizer, text: str, threshold: float = 0.5):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    p_fake = float(probs[0])
    p_real = float(probs[1])
    score = round(p_real * 100, 2)
    label = "Real ✅" if p_real >= threshold else "Fake ❌"
    return label, score, p_fake, p_real


# ============================================================
# UI helpers
# ============================================================
def credibility_band(score: float) -> Tuple[str, str]:
    if score >= 70:
        return "Highly Credible", "good"
    elif score >= 40:
        return "Moderately Credible", "warn"
    return "Low Credibility", "bad"


def render_gauge(score: float) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, 40], "color": "#ff4b4b"},
                    {"range": [40, 70], "color": "#ffa600"},
                    {"range": [70, 100], "color": "#00cc96"},
                ],
            },
            title={"text": "Credibility Score"},
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_stat_card(title: str, value: str, tone: str = "good") -> None:
    tone_cls = {
        "good": "metric-good",
        "warn": "metric-warn",
        "bad": "metric-bad",
    }.get(tone, "metric-good")

    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">{title}</div>
  <div class="metric-value {tone_cls}">{value}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_block(p_real: float, p_fake: float) -> None:
    real_pct = max(0.0, min(100.0, p_real * 100))
    fake_pct = max(0.0, min(100.0, p_fake * 100))


    st.write(f"Real probability: {p_real:.2%}")
    st.write(f"Fake probability: {p_fake:.2%}")

    st.markdown(
        f"""
<div class="card-soft">
  <div class="section-title">Probability Breakdown</div>
  <div class="section-sub">Model confidence for both classes.</div>

  <div class="muted-mini">Real probability</div>
  <div class="prob-track" style="margin:6px 0 10px 0;">
    <div class="prob-real" style="width:{real_pct:.2f}%;"></div>
  </div>
  <div class="muted-mini" style="margin-bottom:14px;"><b>{real_pct:.2f}%</b></div>

  <div class="muted-mini">Fake probability</div>
  <div class="prob-track" style="margin:6px 0 10px 0;">
    <div class="prob-fake" style="width:{fake_pct:.2f}%;"></div>
  </div>
  <div class="muted-mini"><b>{fake_pct:.2f}%</b></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_history_summary(df: pd.DataFrame) -> None:
    total = len(df)
    text_n = int((df["input_type"] == "TEXT").sum()) if not df.empty else 0
    url_n = int((df["input_type"] == "URL").sum()) if not df.empty else 0
    avg_score = float(df["score"].mean()) if not df.empty else 0.0

    st.markdown(
        f"""
<div class="stat-grid">
  <div class="stat-card-pro">
    <div class="stat-label">Total Analyses</div>
    <div class="stat-value">{total}</div>
    <div class="stat-note">All saved records</div>
  </div>
  <div class="stat-card-pro">
    <div class="stat-label">Text / URL Checks</div>
    <div class="stat-value">{text_n} / {url_n}</div>
    <div class="stat-note">By main input mode</div>
  </div>
  <div class="stat-card-pro">
    <div class="stat-label">Average Score</div>
    <div class="stat-value">{avg_score:.2f}</div>
    <div class="stat-note">Across history records</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str) -> None:
    st.markdown(
        f"""
<div class="card-soft">
  <div class="section-title">{title}</div>
  <div class="section-sub">{message}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def build_text_report(
    mode: str,
    label: str,
    score: float,
    source: str,
    extra: Dict[str, str | float],
) -> str:
    lines = [
        "Fake News Credibility Checker - Analysis Report",
        "=" * 50,
        f"Mode: {mode}",
        f"Source: {source}",
        f"Label: {label}",
        f"Score: {score:.2f}%",
        "",
        "Details:",
    ]
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"Generated at: {datetime.utcnow().isoformat()} UTC")
    return "\n".join(lines)


# ============================================================
# Help chatbot
# ============================================================
def help_bot_answer(msg: str) -> str:
    m = (msg or "").strip().lower()
    rules = [
        (["threshold", "slider"], "The threshold controls the text decision: if P(real) is greater than or equal to the threshold, the article is marked as Real."),
        (["score", "credibility"], "In text mode, the credibility score is based on P(real) multiplied by 100. In URL mode, the score comes from the credibility engine."),
        (["url", "source"], "URL mode evaluates source credibility using author verification, publisher transparency, and source corroboration."),
        (["csv", "batch"], "CSV batch mode lets you upload a file, choose the text column, run predictions, and download the results."),
        (["remember", "auto", "login"], f"Remember me stores a session cookie for {SESSION_DAYS} days."),
        (["reset", "forgot", "password"], "Forgot password sends a verification code by email. Then you verify the code and set a new password."),
        (["history"], "History stores your recent checks. You can filter the records and export them to CSV."),
    ]
    for keys, ans in rules:
        if any(k in m for k in keys):
            return ans
    return "Ask me about text analysis, URL credibility checks, CSV batch mode, login, reset password, or history."


def page_help() -> None:
    st.markdown("### 💬 Help Assistant")
    st.caption("Ask how to use the system.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.write(content)

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        st.session_state.chat.append(("assistant", help_bot_answer(prompt)))
        st.rerun()


# ============================================================
# Auth UI
# ============================================================
def auth_gate() -> bool:
    restore_session_from_cookie()

    if st.session_state.get("user_email"):
        return True

    st.markdown(
        """
<div class="navwrap">
  <div class="navtitle">📰 Fake News Credibility Checker</div>
  <div class="navsub">AI-powered text analysis and source credibility evaluation</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.3, 0.9], gap="large")

    with left:
        st.markdown(
            """
<div class="hero">
  <div class="hero-kicker">Hybrid AI Credibility Platform</div>
  <div class="hero-title">Professional credibility analysis for news content</div>
  <div class="hero-sub">
    This system combines a machine learning text classifier with a source credibility engine
    to help users evaluate online news content in a more structured and practical way.
  </div>
  <div style="margin-top:16px;">
    <span class="hero-badge">Machine Learning</span>
    <span class="hero-badge">Source Verification</span>
    <span class="hero-badge">History Tracking</span>
    <span class="hero-badge">CSV Batch Analysis</span>
    <span class="hero-badge">Account Security</span>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        a1, a2, a3 = st.columns(3, gap="small")
        with a1:
            render_stat_card("Core Engine", "Ready", "good")
        with a2:
            render_stat_card("User Access", "Secure", "good")
        with a3:
            render_stat_card("Session", f"{SESSION_DAYS} days", "warn")

    with right:
        st.markdown(
            """
<div class="card">
  <div class="section-title">Account Access</div>
  <div class="section-sub">Sign in, create an account, or reset your password.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        is_dark = (get_theme() == "dark")
        dark_toggle = st.toggle("Dark mode", value=is_dark, key="auth_theme_toggle")
        if dark_toggle != is_dark:
            set_theme("dark" if dark_toggle else "light")
            st.rerun()

        st.write("")
        st.markdown('<div class="auth-switch">', unsafe_allow_html=True)

        if "auth_view" not in st.session_state:
            st.session_state.auth_view = "Login"

        if st.session_state.get("go_to_login"):
            st.session_state["auth_view"] = "Login"
            st.session_state["go_to_login"] = False
            st.session_state["register_success"] = True

        auth_view = st.radio(
            "Account",
            ["Login", "Register", "Forgot password"],
            horizontal=True,
            key="auth_view",
            label_visibility="collapsed",
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        if st.session_state.get("register_success"):
          st.success("Account created successfully. You can now log in.")
          st.session_state["register_success"] = False

        if auth_view == "Login":
            st.markdown(
                """
<div class="card-lite">
  <div class="section-title">Welcome back</div>
  <div class="section-sub">Sign in to continue to your dashboard.</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            token = cookie_manager().get("remember_token") or ""
            remembered_email = verify_remember_token(token) if token else ""

            if remembered_email and "auth_login_email" not in st.session_state:
                st.session_state["auth_login_email"] = remembered_email

            email = st.text_input(
                  "Email",
                  key="auth_login_email",
                  placeholder="name@example.com"
              )
            pw = st.text_input("Password", type="password", key="auth_login_pw")
            remember = st.checkbox(f"Remember me for {SESSION_DAYS} days", value=True, key="auth_remember")

            if st.button("Login", type="primary", use_container_width=True, key="auth_login_btn"):
                rate_limit_gate("login")
                email_norm = (email or "").lower().strip()
                user = get_user(email_norm)

                if not email_norm or "@" not in email_norm:
                    st.error("Please enter a valid email.")
                elif not pw.strip():
                    st.error("Please enter your password.")
                elif not user:
                    st.error("User not found.")
                else:
                    _, stored_hash, _ = user
                    if pbkdf2_verify_password(pw, stored_hash, get_app_secret()):
                        st.session_state.user_email = email_norm
                        update_last_login(email_norm)
                        if remember:
                            create_session(email_norm, days_valid=SESSION_DAYS)
                        else:
                            delete_session()
                        st.success("Logged in successfully.")
                        st.rerun()
                    else:
                        st.error("Incorrect password.")

        elif auth_view == "Register":
            st.markdown(
                """
<div class="card-lite">
  <div class="section-title">Create a new account</div>
  <div class="section-sub">Set up your profile to use text checks, URL checks, and batch analysis.</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            display = st.text_input("Display name (optional)", key="auth_reg_display", placeholder="Your name")
            email2 = st.text_input("Email", key="auth_reg_email", placeholder="name@example.com")
            pw2 = st.text_input("Password (minimum 6 characters)", type="password", key="auth_reg_pw1")
            pw3 = st.text_input("Confirm password", type="password", key="auth_reg_pw2")

            if st.button("Create account", type="primary", use_container_width=True, key="auth_reg_btn"):
              rate_limit_gate("register")
              email2_norm = (email2 or "").lower().strip()

              if not email2_norm or "@" not in email2_norm:
                  st.error("Please enter a valid email.")
              elif len((pw2 or "").strip()) < 6:
                  st.error("Password must be at least 6 characters.")
              elif pw2 != pw3:
                  st.error("Passwords do not match.")
              else:
                  try:
                      register_user(email2_norm, pw2, display_name=display)
                      st.session_state["go_to_login"] = True
                      st.rerun()
                  except sqlite3.IntegrityError:
                      st.error("This email is already registered.")
                  except Exception as e:
                      st.error("Registration failed.")
                      st.code(str(e))
                

        elif auth_view == "Forgot password":
            cfg = smtp_config()
            ready = smtp_is_ready(cfg)

            st.markdown(
                """
<div class="card-lite">
  <div class="section-title">Reset your password</div>
  <div class="section-sub">A verification code will be sent to your email address if SMTP is configured.</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            st.caption(f"The verification code expires in {RESET_CODE_TTL_MIN} minutes.")

            if not ready:
                st.warning("SMTP is not configured in secrets.toml, so email sending is currently disabled.")

            reset_email = st.text_input("Account email", key="reset_email", placeholder="name@example.com")

            if st.button("Send verification code", use_container_width=True, key="reset_send_code"):
                rate_limit_gate("reset_send", seconds=RESET_COOLDOWN_SEC)

                email_norm = (reset_email or "").lower().strip()
                if not email_norm or "@" not in email_norm:
                    st.error("Please enter a valid email.")
                else:
                    user = get_user(email_norm)
                    if not user:
                        st.error("No account was found for this email.")
                    else:
                        code = create_reset_code(email_norm)
                        if ready:
                            try:
                                send_reset_email(email_norm, code)
                                st.success("Verification code sent successfully. Please check your inbox.")
                            except Exception as e:
                                st.error("Email delivery failed. Please try again later or contact support.")
                                
                        else:
                            st.error("SMTP is not configured, so password reset email is unavailable.")

            st.write("")
            st.markdown("**Verify code and set a new password**")
            code_in = st.text_input("Verification code", key="reset_code")
            newp1 = st.text_input("New password", type="password", key="reset_new1")
            newp2 = st.text_input("Confirm new password", type="password", key="reset_new2")

            if st.button("Reset password", type="primary", use_container_width=True, key="reset_apply"):
                rate_limit_gate("reset_apply")

                email_norm = (reset_email or "").lower().strip()

                if not email_norm or "@" not in email_norm:
                    st.error("Please enter your account email above.")
                    st.stop()

                if not verify_reset_code(email_norm, code_in):
                    st.error("Invalid or expired verification code.")
                    st.stop()

                if newp1 != newp2:
                    st.error("Passwords do not match.")
                    st.stop()

                if len((newp1 or "").strip()) < 6:
                    st.error("Password must be at least 6 characters.")
                    st.stop()

                update_password(email_norm, newp1.strip())
                st.success("Password updated successfully. You can now log in.")
                st.session_state.auth_view = "Login"

    return False


# ============================================================
# Pages
# ============================================================
def page_checker(model, vectorizer) -> None:
    email = st.session_state.user_email

    st.markdown("### 🔎 Analysis Workspace")
    st.caption(
        "Use the workspace below to run text classification, source credibility checks, or batch analysis on multiple articles."
    )
    
    st.caption("This tool provides an estimation and should not be used as the sole fact-checking authority.")

    st.markdown(
        """
<div class="hero" style="padding:22px 24px; border-radius:22px;">
  <div class="hero-kicker">Professional Analysis Console</div>
  <div class="hero-title" style="font-size:26px;">Evaluate content credibility through multiple analysis modes</div>
  <div class="hero-sub">
    This workspace allows you to analyze article text, inspect source credibility through URLs,
    and process multiple records through CSV batch mode. The goal is to provide a structured,
    practical, and presentation-ready environment for fake news credibility assessment.
  </div>
  <div style="margin-top:14px;">
    <span class="hero-badge">Text Classification</span>
    <span class="hero-badge">Source Analysis</span>
    <span class="hero-badge">Batch Processing</span>
    <span class="hero-badge">Confidence Scoring</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    threshold = st.slider(
        "Decision threshold for text mode (P(real) ≥ threshold ⇒ Real)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        key="thr",
    )

    st.markdown(
        f"""
<span class="badge">Threshold: <b>{threshold:.2f}</b></span>
<span class="badge">Max text length: <b>{MAX_TEXT_CHARS:,}</b></span>
<span class="badge">Auto-login: <b>{SESSION_DAYS} days</b></span>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Input type",
        ["Paste Text", "Use URL", "Upload CSV (batch)"],
        horizontal=True,
        key="mode",
    )

    show_probs = st.checkbox("Show raw probabilities in text mode", value=False, key="show_probs")

    if mode == "Paste Text":
        left, right = st.columns([1.35, 0.85], gap="large")

        with left:
            st.markdown(
                """
<div class="card-lite">
  <div class="section-title">Text Analysis</div>
  <div class="section-sub">Paste the article text below to evaluate whether the content appears real or fake.</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            txt = st.text_area("Paste article text", height=280, key="txt")

            if not txt.strip():
                st.markdown(
                    """
<div class="card-soft">
  <div class="section-title">Ready for Analysis</div>
  <div class="section-sub">
    Paste article text into the field above and click <b>Run text analysis</b> to generate:
    <br><br>
    • a final prediction<br>
    • a credibility score<br>
    • a confidence band<br>
    • a probability breakdown<br>
    • a downloadable report
  </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button("Run text analysis", type="primary", use_container_width=True, key="btn_txt"):
                rate_limit_gate("check_text")

                if not txt.strip():
                    st.warning("Please paste some text first.")
                    st.stop()

                if len(txt) > MAX_TEXT_CHARS:
                    st.error(f"Text is too large. Maximum allowed length is {MAX_TEXT_CHARS:,} characters.")
                    st.stop()

                label, score, p_fake, p_real = predict(model, vectorizer, txt, threshold=threshold)
                band, tone = credibility_band(score)
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]

                vec_input = vectorizer.transform([clean_text(txt)])
                indices = vec_input.nonzero()[1]

                top_features = sorted(
                    [(feature_names[i], coefs[i]) for i in indices],
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]

                st.write("")
                r1, r2 = st.columns([1, 1], gap="large")

                with r1:
                    render_stat_card("Prediction", label, tone)
                    st.write("")
                    render_stat_card("Credibility Score", f"{score:.2f}%", tone)
                    st.write("")
                    render_stat_card("Confidence Band", band, tone)

                with r2:
                    render_gauge(score)

                st.write("")
                render_probability_block(p_real, p_fake)
                st.write("")
                st.markdown("### 🔍 Key Influencing Words")

                for word, weight in top_features:
                    color = "🟢" if weight > 0 else "🔴"
                    st.write(f"{color} {word} ({weight:.4f})")

                st.write("")
                st.markdown(
                    f"""
<div class="result-panel">
  <div class="section-title">Analysis Summary</div>
  <div class="section-sub">
    The submitted article has been processed through the text classification pipeline and evaluated using the active decision threshold.
    The final outcome suggests a classification of <b>{label}</b>, with an overall credibility score of <b>{score:.2f}%</b>.
    Based on the score range, the content falls under the <b>{band}</b> category.
    <br><br>
    Additional model outputs indicate a real probability of <b>{p_real:.4f}</b>, a fake probability of <b>{p_fake:.4f}</b>,
    and a threshold setting of <b>{threshold:.2f}</b>.
  </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")
                if tone == "good":
                    st.success(
                        f"This article is classified as {band.lower()} based on the current text classification model."
                    )
                elif tone == "warn":
                    st.warning(
                        f"This article falls into the {band.lower()} range and should be reviewed carefully."
                    )
                else:
                    st.error(
                        f"This article falls into the {band.lower()} range and may contain misleading content."
                    )
                
                st.info(
                    "Institutional websites such as universities, government domains, or research organizations "
                    "may not always follow standard news article structures. As a result, the system may return "
                    "moderate or lower credibility scores even when the source itself is trustworthy."
                )

                report_txt = build_text_report(
                    mode="TEXT",
                    label=label,
                    score=score,
                    source="manual text input",
                    extra={
                        "Credibility Band": band,
                        "Probability Real": round(p_real, 4),
                        "Probability Fake": round(p_fake, 4),
                        "Threshold Used": round(threshold, 2),
                    },
                )

                st.download_button(
                    "Download analysis report",
                    data=report_txt.encode("utf-8"),
                    file_name="text_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_text_report",
                )

                if show_probs:
                    st.write("")
                    st.json(
                        {
                            "P_fake": round(p_fake, 4),
                            "P_real": round(p_real, 4),
                            "threshold_used": round(threshold, 2),
                        }
                    )

                save_history(email, "TEXT", "manual", label, score, p_fake, p_real, threshold)

        with right:
            st.markdown(
                """
<div class="card">
  <div class="section-title">Analysis Guide</div>
  <div class="section-sub">What happens in text mode</div>
  <div class="muted-mini">1. Input text is validated</div>
  <div class="muted-mini" style="margin-top:6px;">2. Text is cleaned and normalized</div>
  <div class="muted-mini" style="margin-top:6px;">3. TF-IDF features are generated</div>
  <div class="muted-mini" style="margin-top:6px;">4. The ML model calculates probabilities</div>
  <div class="muted-mini" style="margin-top:6px;">5. A threshold-based final label is assigned</div>
  <div class="soft-divider"></div>
  <div class="section-title">Best Use Cases</div>
  <div class="muted-mini">• pasted article text</div>
  <div class="muted-mini" style="margin-top:6px;">• extracted article content</div>
  <div class="muted-mini" style="margin-top:6px;">• dataset testing through CSV mode</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    elif mode == "Use URL":
        left, right = st.columns([1.25, 0.95], gap="large")

        with left:
            st.markdown(
                """
<div class="card-lite">
  <div class="section-title">Source Credibility Analysis</div>
  <div class="section-sub">Paste a news URL to evaluate credibility signals related to source quality and transparency.</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            url = st.text_input("Paste URL", placeholder="https://...", key="url")
            st.caption("This mode evaluates author verification, publisher transparency, and source corroboration.")

            if st.button("Fetch and analyze URL", type="primary", use_container_width=True, key="btn_url"):
                rate_limit_gate("check_url")

                if not url.strip():
                    st.warning("Please paste a URL first.")
                    st.stop()

                try:
                    with st.spinner("Analyzing source credibility..."):
                        result = evaluate_source(url)

                    final_score = float(result["final_score"])
                    band, tone = credibility_band(final_score)

                    st.write("")
                    r1, r2 = st.columns([1, 1], gap="large")

                    with r1:
                        render_stat_card("Label", str(result["label"]), tone)
                        st.write("")
                        render_stat_card("Final Score", f"{final_score:.2f}", tone)
                        st.write("")
                        render_stat_card("Confidence Band", band, tone)

                    with r2:
                        render_gauge(final_score)

                    st.write("")
                    st.markdown("### Score Breakdown")
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown(
                            f"""
<div class="card-soft">
  <div class="section-title">Author Verification</div>
  <div class="big-label">{result["author_score"]}</div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"""
<div class="card-soft">
  <div class="section-title">Publisher Transparency</div>
  <div class="big-label">{result["transparency_score"]}</div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            f"""
<div class="card-soft">
  <div class="section-title">Source Corroboration</div>
  <div class="big-label">{result["corroboration_score"]}</div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.write("")
                    st.markdown("### Article Title")
                    st.markdown(
                        f"""
<div class="result-panel">
  <div>{result["title"]}</div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write("")
                    st.markdown(
                        f"""
<div class="result-panel">
  <div class="section-title">URL Analysis Summary</div>
  <div class="section-sub">
    Final score: <b>{final_score:.2f}</b><br>
    Confidence band: <b>{band}</b><br>
    Author verification: <b>{result["author_score"]}</b><br>
    Publisher transparency: <b>{result["transparency_score"]}</b><br>
    Source corroboration: <b>{result["corroboration_score"]}</b>
  </div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write("")
                    if tone == "good":
                        st.success("This source appears relatively credible based on the current credibility indicators.")
                    elif tone == "warn":
                        st.warning("This source shows mixed credibility signals and should be evaluated carefully.")
                    else:
                        st.error("This source shows weak credibility indicators and may be suspicious.")

                    report_txt = build_text_report(
                        mode="URL",
                        label=str(result["label"]),
                        score=final_score,
                        source=url,
                        extra={
                            "Credibility Band": band,
                            "Author Verification": result["author_score"],
                            "Publisher Transparency": result["transparency_score"],
                            "Source Corroboration": result["corroboration_score"],
                            "Article Title": result["title"],
                        },
                    )

                    st.download_button(
                        "Download URL analysis report",
                        data=report_txt.encode("utf-8"),
                        file_name="url_analysis_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="dl_url_report",
                    )

                    with st.expander("Show full technical details"):
                        st.json(result)

                    save_history(
                        email,
                        "URL",
                        url,
                        result["label"],
                        final_score,
                        0.0,
                        0.0,
                        threshold,
                    )

                except Exception as e:
                  st.error("Could not fetch or analyze this URL. The site may block automated access or limit scraping.")
                  st.code(str(e))

        with right:
            st.markdown(
                """
<div class="card">
  <div class="section-title">URL Mode Overview</div>
  <div class="section-sub">How source analysis works</div>
  <div class="muted-mini">• URL content is fetched and parsed</div>
  <div class="muted-mini" style="margin-top:6px;">• metadata and article signals are examined</div>
  <div class="muted-mini" style="margin-top:6px;">• author verification is checked</div>
  <div class="muted-mini" style="margin-top:6px;">• transparency signals are scored</div>
  <div class="muted-mini" style="margin-top:6px;">• corroboration contributes to the final score</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    else:
        left, right = st.columns([1.3, 0.9], gap="large")

        with left:
            st.markdown(
                """
<div class="card-lite">
  <div class="section-title">CSV Batch Analysis</div>
  <div class="section-sub">Upload a CSV file, specify the text column, and generate predictions for multiple rows at once.</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            up = st.file_uploader("Upload CSV", type=["csv"], key="csv")
            text_col = st.text_input("Text column name", value="text", key="csv_col")

            df_up = None
            if up is not None:
                try:
                    try:
                        df_up = pd.read_csv(up, encoding="utf-8")
                    except UnicodeDecodeError:
                        up.seek(0)
                        df_up = pd.read_csv(up, encoding="latin-1")

                    st.write("Preview:")
                    st.dataframe(df_up.head(10), use_container_width=True)

                except Exception as e:
                    st.error("Could not read the CSV file.")
                    st.code(str(e))
                    df_up = None

            if st.button("Run batch analysis", type="primary", use_container_width=True, key="btn_batch"):
                rate_limit_gate("check_batch")

                if df_up is None:
                    st.warning("Please upload a CSV file first.")
                    st.stop()

                if text_col not in df_up.columns:
                    st.error(f"Column '{text_col}' was not found. Available columns: {list(df_up.columns)}")
                    st.stop()

                df = df_up.copy()
                df[text_col] = df[text_col].astype(str).fillna("")
                df["clean_text"] = df[text_col].apply(clean_text)

                vecs = vectorizer.transform(df["clean_text"])
                probs = model.predict_proba(vecs)

                df["P_fake"] = probs[:, 0].astype(float)
                df["P_real"] = probs[:, 1].astype(float)
                df["Credibility_score"] = (df["P_real"] * 100).round(2)
                df["Prediction"] = df["P_real"].apply(lambda p: "Real ✅" if p >= threshold else "Fake ❌")
                df["Threshold_used"] = threshold

                st.success("Batch prediction completed successfully.")
                st.dataframe(
                    df[[text_col, "Prediction", "Credibility_score", "P_fake", "P_real", "Threshold_used"]].head(50),
                    use_container_width=True,
                )

                st.download_button(
                    "Download results CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="credibility_results.csv",
                    mime="text/csv",
                    key="dl_results",
                    use_container_width=True,
                )

                save_history(
                    email,
                    "CSV",
                    f"rows={len(df)} col={text_col}",
                    "batch",
                    float(df["Credibility_score"].mean()),
                    float(df["P_fake"].mean()),
                    float(df["P_real"].mean()),
                    threshold,
                )

        with right:
            st.markdown(
                """
<div class="card">
  <div class="section-title">Batch Mode Notes</div>
  <div class="section-sub">What this mode is for</div>
  <div class="muted-mini">• evaluate many rows at once</div>
  <div class="muted-mini" style="margin-top:6px;">• compare predictions across a dataset</div>
  <div class="muted-mini" style="margin-top:6px;">• export the final results as CSV</div>
  <div class="soft-divider"></div>
  <div class="section-title">Recommended CSV Format</div>
  <div class="muted-mini">Include one text column such as <b>text</b>.</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def page_history() -> None:
    st.markdown("### 🕘 Analysis History")
    st.caption("Review your recent activity, filter records, export them, or clear your saved history.")

    email = st.session_state.user_email
    dfh = load_history(email, limit=500)

    if dfh.empty:
        render_empty_state(
            "No analysis history yet",
            "Run your first text, URL, or CSV check to populate your history dashboard.",
        )
        return

    dfh["created_at"] = pd.to_datetime(dfh["created_at"], errors="coerce")
    render_history_summary(dfh)
    st.write("")

    top_a, top_b = st.columns([1, 0.3])
    with top_b:
        if st.button("Clear history", use_container_width=True, key="clear_history_btn"):
            clear_history(email)
            st.success("History cleared successfully.")
            st.rerun()

    c1, c2, c3 = st.columns([1, 1.2, 0.8], gap="large")
    with c1:
        t_filter = st.multiselect(
            "Filter by input type",
            options=["TEXT", "URL", "CSV"],
            default=["TEXT", "URL", "CSV"],
        )
    with c2:
        search = st.text_input("Search by source, prediction, or type", value="", key="hist_search")
    with c3:
        st.write("")
        if st.button("Clear search", use_container_width=True):
            st.session_state.hist_search = ""
            st.rerun()

    df = dfh[dfh["input_type"].isin(t_filter)]

    if search.strip():
        q = search.strip()
        df = df[
            df["source"].astype(str).str.contains(q, case=False, na=False)
            | df["prediction"].astype(str).str.contains(q, case=False, na=False)
            | df["input_type"].astype(str).str.contains(q, case=False, na=False)
        ]

    if not df.empty:
        df = df.copy()
        df["score"] = df["score"].round(2)

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download history CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="history.csv",
        mime="text/csv",
        key="dl_hist",
        use_container_width=True,
    )


def page_account() -> None:
    st.markdown("### 👤 Account Settings")
    st.caption("Manage your profile information and security settings.")

    email = st.session_state.user_email
    user = get_user(email)
    current_display = (user[2] if user else None) or ""

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">Profile</div>
  <div class="section-sub">Update your display name for a more personalized dashboard.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        new_display = st.text_input("Display name", value=current_display, key="acc_display")
        if st.button("Save profile", type="primary", use_container_width=True, key="acc_save_profile"):
            rate_limit_gate("save_profile")
            update_display_name(email, new_display)
            st.success("Profile updated successfully.")
            st.rerun()

    with c2:
        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">Session</div>
  <div class="section-sub">Sign out securely from the current device.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Logout", use_container_width=True, key="acc_logout_btn"):
          delete_session()
          st.rerun()

    st.markdown("---")
    st.markdown("**Change password**")

    old_pw = st.text_input("Current password", type="password", key="acc_old_pw")
    new_pw1 = st.text_input("New password (minimum 6 characters)", type="password", key="acc_new_pw1")
    new_pw2 = st.text_input("Confirm new password", type="password", key="acc_new_pw2")

    if st.button("Update password", type="primary", use_container_width=True, key="acc_update_pw"):
        rate_limit_gate("change_pw")

        row = get_user(email)
        if not row:
            st.error("User not found.")
            st.stop()

        _, stored_hash, _ = row
        if not pbkdf2_verify_password(old_pw, stored_hash, get_app_secret()):
            st.error("Current password is incorrect.")
            st.stop()

        if new_pw1 != new_pw2:
            st.error("The new passwords do not match.")
            st.stop()

        if len(new_pw1.strip()) < 6:
            st.error("The new password must be at least 6 characters long.")
            st.stop()

        update_password(email, new_pw1.strip())
        st.success("Password updated successfully.")


def page_about() -> None:
    st.markdown("### ℹ️ About the System")
    st.caption("Overview of the project, its architecture, capabilities, and practical value.")

    st.markdown(
        """
<div class="about-hero">
  <div class="about-hero-title">Fake News Credibility Checker</div>
  <div class="about-hero-sub">
    The Fake News Credibility Checker is a hybrid analysis system that combines machine learning
    text classification with source credibility evaluation. The platform is designed to provide
    a more complete and realistic understanding of online news reliability by examining both
    the content and the origin of information.
  </div>

  <div style="margin-top:14px;">
    <span class="about-pill">Machine Learning</span>
    <span class="about-pill">NLP</span>
    <span class="about-pill">Source Credibility</span>
    <span class="about-pill">Interactive Dashboard</span>
    <span class="about-pill">Secure Platform</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="card">
  <div class="section-title">System Overview</div>
  <div class="section-sub">
    The Fake News Credibility Checker is a capstone project developed to evaluate online news credibility
    through a dual-analysis strategy. Instead of relying only on article text, the platform combines
    machine learning-based text evaluation with source-focused credibility assessment.
    <br><br>
    This design provides a broader and more practical understanding of whether digital news content appears
    trustworthy, suspicious, or in need of further verification.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">1. Core Purpose</div>
  <div class="section-sub">
    The main goal of the application is to support users in evaluating online news in a structured,
    accessible, and user-friendly environment. The platform is intended to function as a credibility-support tool,
    helping users inspect both the article text and the reliability signals of the source behind it.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">2. Text-Based Analysis</div>
  <div class="section-sub">
    In text mode, users can paste article content directly into the system. The content is cleaned,
    normalized, transformed with TF-IDF vectorization, and then evaluated through a trained machine learning model.
    The system returns a final prediction, a credibility score, and supporting probability values.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">3. Source Credibility Analysis</div>
  <div class="section-sub">
    In URL mode, the system evaluates the reliability of a webpage using source-oriented credibility indicators.
    These include author verification, publisher transparency, and source corroboration. The final result offers
    a more complete judgment than plain text classification alone.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">4. Batch Processing and Productivity</div>
  <div class="section-sub">
    The platform supports CSV batch analysis, allowing multiple text records to be processed in one run.
    This functionality improves scalability, supports experimentation with datasets, and makes the system
    more practical for demonstrations and academic evaluation.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">5. User Features and Security</div>
  <div class="section-sub">
    The application includes account registration, secure login, password reset with verification code,
    session persistence with remember-me support, and local history storage. These features make the project
    more complete and closer to a realistic digital product environment.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
<div class="card-lite">
  <div class="section-title">6. User Experience and Visualization</div>
  <div class="section-sub">
    The interface has been designed to present analysis results clearly and professionally. It includes
    dashboard summaries, recent activity panels, score cards, probability breakdowns, report downloads,
    and dark mode support in order to improve both usability and presentation quality.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    f1, f2, f3 = st.columns(3, gap="large")

    with f1:
        st.markdown(
            """
<div class="card-soft">
  <div class="section-title">Analysis Features</div>
  <div class="section-sub">
    • text credibility analysis<br>
    • URL-based source evaluation<br>
    • CSV batch processing<br>
    • credibility score generation<br>
    • probability breakdown display
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            """
<div class="card-soft">
  <div class="section-title">User Features</div>
  <div class="section-sub">
    • account registration<br>
    • secure login and logout<br>
    • password reset flow<br>
    • remember-me sessions<br>
    • account settings update
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            """
<div class="card-soft">
  <div class="section-title">Platform Features</div>
  <div class="section-sub">
    • analysis history storage<br>
    • recent activity dashboard<br>
    • downloadable reports<br>
    • help assistant<br>
    • light and dark mode support
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        """
<div class="card">
  <div class="section-title">Project Value</div>
  <div class="section-sub">
    The strength of the project lies in its hybrid architecture. Instead of evaluating only what the text says,
    the system also examines where the information comes from. This makes the output more informative,
    more realistic, and better aligned with the actual challenge of assessing misinformation in digital environments.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="card-soft">
  <div class="section-title">Future Improvements</div>
  <div class="section-sub">
    Future extensions could include stronger explainability, richer dashboard analytics, expanded source reference lists,
    improved corroboration logic, cloud deployment, and more advanced machine learning models for higher predictive performance.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================
init_db()
update_reference_lists()

theme = get_theme()
inject_css(theme)
apply_pending_cookie_writes()

logged_in = auth_gate()
if not logged_in:
    st.stop()

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Could not load the model or vectorizer. Make sure the .pkl files exist in the /models folder.")
    st.code(str(e))
    st.stop()

user = get_user(st.session_state.user_email)
display = (user[2] if user else None) or ""
label = f"{display} ({st.session_state.user_email})" if display else st.session_state.user_email
stats = get_user_stats(st.session_state.user_email)
recent_df = load_recent_history(st.session_state.user_email, limit=5)
avg_band, avg_tone = credibility_band(stats["avg_score"]) if stats["total"] > 0 else ("No Data", "warn")

st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)

hero_left, hero_right = st.columns([1.7, 1.0], gap="large")

with hero_left:
    st.markdown(
        """
<div class="hero">
  <div class="hero-kicker">Hybrid AI Credibility Platform</div>
  <div class="hero-title">Fake News Credibility Intelligence System</div>
  <div class="hero-sub">
    A professional analysis environment that combines machine learning text classification
    with source credibility evaluation to support more informed assessment of online news content.
  </div>
  <div style="margin-top:16px;">
    <span class="hero-badge">ML Text Classification</span>
    <span class="hero-badge">Source Trust Scoring</span>
    <span class="hero-badge">CSV Batch Analysis</span>
    <span class="hero-badge">User History</span>
    <span class="hero-badge">Secure Authentication</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        f"""
<div class="side-panel">
  <div class="side-panel-title">Signed in</div>
  <div class="side-panel-sub">{label}</div>
  <div class="soft-divider"></div>
  <div style="font-size:13px; color: var(--muted) !important;">Environment</div>
  <div style="margin-top:4px; font-weight:700;">Streamlit + ML + SQLite</div>
  <div style="margin-top:14px; font-size:13px; color: var(--muted) !important;">Authentication</div>
  <div style="margin-top:4px; font-weight:700;">Session persistence enabled</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    is_dark = get_theme() == "dark"
    new_toggle = st.toggle("Dark mode", value=is_dark, key="nav_theme_toggle")
    if new_toggle != is_dark:
        set_theme("dark" if new_toggle else "light")
        st.rerun()

    if st.button("Logout", use_container_width=True, key="top_logout_btn"):
      delete_session()
      st.rerun() 

st.write("")

st.markdown(
    f"""
<div class="stat-grid">
  <div class="stat-card-pro">
    <div class="stat-label">Analysis Modes</div>
    <div class="stat-value">3</div>
    <div class="stat-note">Text, URL, CSV batch</div>
  </div>
  <div class="stat-card-pro">
    <div class="stat-label">Session Duration</div>
    <div class="stat-value">{SESSION_DAYS} days</div>
    <div class="stat-note">Persistent login option</div>
  </div>
  <div class="stat-card-pro">
    <div class="stat-label">Max Text Capacity</div>
    <div class="stat-value">{MAX_TEXT_CHARS:,}</div>
    <div class="stat-note">Character limit per request</div>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

st.write("")

st.markdown("### 📊 Personal Dashboard")
st.caption("Quick overview of your recent activity and usage.")

d1, d2, d3, d4 = st.columns(4, gap="large")

with d1:
    render_stat_card("Total Analyses", str(stats["total"]), "good")
with d2:
    render_stat_card("Text Checks", str(stats["text_count"]), "good")
with d3:
    render_stat_card("URL Checks", str(stats["url_count"]), "warn")
with d4:
    render_stat_card("Average Score", f'{stats["avg_score"]:.2f}', avg_tone)

st.write("")

st.markdown("### 🕘 Recent Activity")
if recent_df.empty:
    render_empty_state(
        "No recent activity",
        "Your latest analyses will appear here once you start using the checker.",
    )
else:
    recent_df["created_at"] = pd.to_datetime(recent_df["created_at"], errors="coerce")
    st.dataframe(recent_df, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

tab_checker, tab_history, tab_account, tab_about, tab_help = st.tabs(
    ["🔎 Checker", "🕘 History", "👤 Account", "ℹ️ About", "💬 Help"]
)

with tab_checker:
    page_checker(model, vectorizer)

with tab_history:
    page_history()

with tab_account:
    page_account()

with tab_about:
    page_about()

with tab_help:
    page_help()