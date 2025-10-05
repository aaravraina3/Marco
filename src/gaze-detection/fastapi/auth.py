"""
Authentication and session management for gaze tracker
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import hashlib
import secrets

SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# In-memory user database (for MVP - replace with real DB later)
users_db = {}

# Session storage: {session_id: {user_id, calibration_data, timestamp}}
sessions_db = {}

def verify_password(plain_password, hashed_password):
    """Verify password using SHA256 (simple for MVP)"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_password_hash(password):
    """Hash password using SHA256 (simple for MVP)"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str):
    if username in users_db:
        return None

    users_db[username] = {
        "username": username,
        "hashed_password": get_password_hash(password),
        "created_at": datetime.now().isoformat()
    }
    return users_db[username]

def authenticate_user(username: str, password: str):
    if username not in users_db:
        return None
    user = users_db[username]
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

def create_session(user_id: str):
    session_id = secrets.token_urlsafe(32)
    sessions_db[session_id] = {
        "user_id": user_id,
        "calibration_data": None,
        "created_at": datetime.now().isoformat()
    }
    return session_id

def get_session(session_id: str):
    return sessions_db.get(session_id)

def save_calibration(session_id: str, calibration_data):
    if session_id in sessions_db:
        sessions_db[session_id]["calibration_data"] = calibration_data
        return True
    return False

def get_calibration(session_id: str):
    session = sessions_db.get(session_id)
    if session:
        return session.get("calibration_data")
    return None
