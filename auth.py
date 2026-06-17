"""
Authentication module for Quality Report LG Sinarmas dashboard.

✅ Works both locally (.env) and on Streamlit Cloud (st.secrets)
✅ Always prioritizes Streamlit Secrets (to fix "password keep 12345" issue)
"""

import os
import time
import hashlib
from typing import Optional
from dotenv import load_dotenv
import streamlit as st

# ======================================================
# Load environment variables (for local development)
# ======================================================
load_dotenv()

# ⚙️ Prefer Streamlit Secrets on Cloud, fallback to .env only if secrets not found
USERNAME = st.secrets.get("USERNAMES") or os.getenv("USERNAMES")
PASSWORD = st.secrets.get("PASSWORDS") or os.getenv("PASSWORDS")

# ======================================================
# Security constants
# ======================================================
COOKIE_KEY = "session_token"
COOKIE_EXPIRY_DAYS = 30

# Generate cookie hash
COOKIE_VALUE = (
    hashlib.sha256(f"{USERNAME}:{PASSWORD}".encode()).hexdigest()
    if USERNAME and PASSWORD
    else None
)

# ======================================================
# Authentication logic
# ======================================================
def validate_login(username: str, password: str) -> bool:
    """
    Validate user credentials against stored values.
    Supports plain text or hashed password if needed.
    """
    if not USERNAME or not PASSWORD:
        st.error(
            "⚠️ Authentication credentials not configured.\n\n"
            "Please set USERNAMES and PASSWORDS in Streamlit Secrets or .env."
        )
        return False

    # ✅ Plain text match (default)
    if username == USERNAME and password == PASSWORD:
        return True

    # 🔐 Optional: enable hashed password comparison if you decide to hash later
    # hashed_input = hashlib.sha256(password.encode()).hexdigest()
    # if username == USERNAME and hashed_input == PASSWORD:
    #     return True

    return False


def is_authenticated() -> bool:
    """Check if the user is authenticated."""
    if st.session_state.get("logged_in", False) or st.session_state.get(COOKIE_KEY) == COOKIE_VALUE:
        return True
        
    # Restore session from query params after a page refresh
    if hasattr(st, "query_params") and COOKIE_KEY in st.query_params:
        if st.query_params[COOKIE_KEY] == COOKIE_VALUE:
            st.session_state["logged_in"] = True
            st.session_state[COOKIE_KEY] = COOKIE_VALUE
            return True
            
    return False


def logout() -> None:
    """Log out the user by clearing session state and cookies."""
    for key in ("logged_in", COOKIE_KEY):
        if key in st.session_state:
            del st.session_state[key]
            
    if hasattr(st, "query_params") and COOKIE_KEY in st.query_params:
        del st.query_params[COOKIE_KEY]


def show_login() -> bool:
    """
    Display the login form and handle authentication.
    Returns True if user is authenticated.
    """
    if is_authenticated():
        return True

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%) !important;
    }

    /* Hide hamburger menu & footer on login */
    #MainMenu, footer { visibility: hidden; }

    /* Logo */
    [data-testid="stImage"] img,
    [data-testid="stImage"] {
        background-color: white !important;
        padding: 8px !important;
        border-radius: 8px !important;
    }

    /* Card container */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(99, 102, 241, 0.1) !important;
    }

    /* Input fields */
    .stTextInput > div > div {
        background: rgba(30, 35, 60, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }
    .stTextInput label { color: #8892b0 !important; font-size: 0.78rem !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }

    /* Login button */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Vertical centering spacer
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    _, col_login, _ = st.columns([1.5, 1.5, 1.5])
    with col_login:
        st.markdown("<div style='text-align:center; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        st.image("lgsm_logo.png", width=160)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center; margin-bottom: 2rem;'>
            <h1 style='font-size: 1.8rem; font-weight: 700;
                background: linear-gradient(90deg, #a5b4fc, #818cf8);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; margin-bottom: 0.3rem;'>
                IDB Quality Dashboard
            </h1>
            <p style='color: #64748b; font-size: 0.9rem; margin: 0;'>LG SinarMas — Sign in to continue</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", border=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("🔐  Sign In", use_container_width=True)

            if submit_button:
                if validate_login(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state[COOKIE_KEY] = COOKIE_VALUE
                    
                    if hasattr(st, "query_params"):
                        st.query_params[COOKIE_KEY] = COOKIE_VALUE
                        
                    st.success("✅ Login successful! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
                    return False

    return False


def require_auth(func):
    """Decorator to protect Streamlit pages with authentication."""
    def wrapper(*args, **kwargs):
        if not show_login():
            return
        return func(*args, **kwargs)

    return wrapper
