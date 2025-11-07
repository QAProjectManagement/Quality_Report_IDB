"""
Authentication module for Quality Report LG Sinarmas dashboard.

This module provides authentication functionality for protecting access to
the dashboard. It handles login validation, session management, and cookie-based
authentication persistence.
""" 

import os
import time
import hashlib
from typing import Tuple, Optional

from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Authentication configuration
USERNAME = os.getenv("USERNAMES")
PASSWORD = os.getenv("PASSWORDS")

# Security constants
COOKIE_KEY = "session_token"
COOKIE_EXPIRY_DAYS = 30
COOKIE_VALUE = hashlib.sha256(f"{USERNAME}:{PASSWORD}".encode()).hexdigest()


def validate_login(username: str, password: str) -> bool:
    """
    Validate user credentials against stored values.
    
    Parameters:
    -----------
    username : str
        The username entered by the user
    password : str
        The password entered by the user
        
    Returns:
    --------
    bool
        True if credentials are valid, False otherwise
    """
    if not USERNAME or not PASSWORD:
        st.error("Authentication credentials not configured.")
        return False
        
    return username == USERNAME and password == PASSWORD


def is_authenticated() -> bool:
    """
    Check if the user is authenticated.
    
    Returns:
    --------
    bool
        True if the user is authenticated, False otherwise
    """
    return (
        st.session_state.get("logged_in", False) or 
        st.session_state.get(COOKIE_KEY) == COOKIE_VALUE
    )


def logout() -> None:
    """
    Log out the user by clearing session state and cookies.
    """
    if "logged_in" in st.session_state:
        del st.session_state["logged_in"]
    
    if COOKIE_KEY in st.session_state:
        del st.session_state[COOKIE_KEY]


def show_login() -> bool:
    """
    Display the login interface and handle authentication process.
    
    Returns:
    --------
    bool
        True if the user is authenticated, False otherwise
    """
    # If already authenticated, skip login
    if is_authenticated():
        return True

    # Display login form centered within the page layout
    spacer_left, main_column, spacer_right = st.columns([1, 2, 1])
    with main_column:
        st.image("lgsm_logo.png", width=220)
        st.title("Login - Quality Report LG Sinarmas")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if validate_login(username, password):
                    # Set authentication status
                    st.session_state["logged_in"] = True
                    st.session_state[COOKIE_KEY] = COOKIE_VALUE

                    # Show success message
                    st.success("Login successful! Redirecting to the dashboard...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
                    return False
    
    # Add some information text
    # st.info("Please login to access the quality report dashboard.")
    
    return False


def require_auth(func):
    """
    Decorator to require authentication for a Streamlit page.
    
    Parameters:
    -----------
    func : callable
        The function to be protected by authentication
        
    Returns:
    --------
    callable
        Wrapped function that checks authentication before execution
    """
    def wrapper(*args, **kwargs):
        if not show_login():
            return
        return func(*args, **kwargs)
    return wrapper
