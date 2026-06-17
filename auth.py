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
    # If already authenticated
    if is_authenticated():
        return True

    # Centered login layout
    spacer_left, main_column, spacer_right = st.columns([1, 2, 1])
    with main_column:
        st.image("lgsm_logo.png", width=220)
        st.title("Login - Quality Report LG SinarMas")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

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
