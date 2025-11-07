"""
Authentication module for Quality Report LG Sinarmas dashboard.

Works both locally (.env) and on Streamlit Cloud (st.secrets).
"""

import os
import time
import hashlib
from typing import Optional
from dotenv import load_dotenv
import streamlit as st

# Load .env for local development
load_dotenv()

# Read credentials from Streamlit Secrets (preferred) or .env (fallback)
USERNAME = st.secrets.get("USERNAMES", os.getenv("USERNAMES"))
PASSWORD = st.secrets.get("PASSWORDS", os.getenv("PASSWORDS"))

# Security constants
COOKIE_KEY = "session_token"
COOKIE_EXPIRY_DAYS = 30

# Generate cookie hash based on credentials
if USERNAME and PASSWORD:
    COOKIE_VALUE = hashlib.sha256(f"{USERNAME}:{PASSWORD}".encode()).hexdigest()
else:
    COOKIE_VALUE = None


def validate_login(username: str, password: str) -> bool:
    """
    Validate user credentials against stored values.
    Supports plain text or hashed password if needed.
    """
    if not USERNAME or not PASSWORD:
        st.error("⚠️ Authentication credentials not configured. "
                 "Please set USERNAMES and PASSWORDS in Streamlit Secrets or .env.")
        return False

    # Direct match (default)
    if password == PASSWORD:
        return username == USERNAME

    # Optional: Uncomment if you use hashed passwords
    # hashed_input = hashlib.sha256(password.encode()).hexdigest()
    # return username == USERNAME and hashed_input == PASSWORD

    return False


def is_authenticated() -> bool:
    """
    Check if the user is authenticated.
    """
    return (
        st.session_state.get("logged_in", False)
        or st.session_state.get(COOKIE_KEY) == COOKIE_VALUE
    )


def logout() -> None:
    """
    Log out the user by clearing session state and cookies.
    """
    for key in ("logged_in", COOKIE_KEY):
        if key in st.session_state:
            del st.session_state[key]


def show_login() -> bool:
    """
    Display the login interface and handle authentication.
    Returns True if user is authenticated.
    """
    # Already authenticated
    if is_authenticated():
        return True

    # Login form
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
                    st.success("✅ Login successful! Redirecting to the dashboard...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
                    return False

    return False


def require_auth(func):
    """
    Decorator to protect Streamlit pages with authentication.
    """
    def wrapper(*args, **kwargs):
        if not show_login():
            return
        return func(*args, **kwargs)

    return wrapper
