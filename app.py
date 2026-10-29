import os
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "fruit_quality_model.keras")
DB_PATH = os.path.join(APP_DIR, "fruit_quality_users.db")

st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --green:#18d994; --green2:#00b981; --bg:#061019; --panel:#0c1b27; --muted:#9db0bf; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #12352e 0%, #061019 38%, #050c13 100%); color:#f3f8fb; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.03em; }
.block-container { max-width:1200px; padding-top:1.8rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#07131d; border-right:1px solid #193342; }
[data-testid="stSidebar"] * { color:#e9f3f7; }
.hero { background:linear-gradient(120deg,rgba(0,185,129,.22),rgba(11,28,41,.9)); border:1px solid #1b5548; border-radius:22px; padding:28px 30px; margin:8px 0 22px; }
.eyebrow { color:#43e6ae; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; }
.hero h1 { font-size:2.3rem; margin:.35rem 0 .4rem; }
.subtle { color:#a9bdc9; }
.metric-card { background:linear-gradient(145deg,#102433,#0a1722); border:1px solid #1b3545; border-radius:16px; padding:18px 20px; min-height:120px; }
.metric-label { color:#9fb3c0; font-size:.83rem; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700; margin-top:8px; }
.panel { background:rgba(10,25,37,.86); border:1px solid #1a3545; border-radius:18px; padding:22px; }
.pill { display:inline-block; border-radius:999px; padding:5px 11px; font-size:.8rem; font-weight:700; }
.fresh { color:#4bf0b0; background:rgba(24,217,148,.12); border:1px solid rgba(24,217,148,.3); }
.rotten { color:#ff8c91; background:rgba(255,82,92,.12); border:1px solid rgba(255,82,92,.3); }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#00b981,#20dfa0); border:0; color:#032116; font-weight:700; border-radius:10px; min-height:2.8rem; }
div.stButton > button { border-radius:10px; }
div[data-testid="stFileUploader"] { background:rgba(8,24,35,.7); border:1px dashed #286451; border-radius:15px; padding:12px; }
div[data-testid="stDataFrame"] { border:1px solid #1a3545; border-radius:12px; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Database / authentication ----------
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        salt BLOB NOT NULL,
        password_hash BLOB NOT NULL,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    con.commit()
    return con

def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return salt, digest

def register_user(username, email, password):
    username, email = username.strip(), email.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    salt, digest = hash_password(password)
    con = db()
    try:
        con.execute("INSERT INTO users(username,email,salt,password_hash,created_at) VALUES(?,?,?,?,?)",
                    (username, email, salt, digest, datetime.now().isoformat(timespec="seconds")))
        con.commit()
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    finally:
        con.close()

def authenticate(username_or_email, password):
    con = db()
    row = con.execute("SELECT id, username, salt, password_hash FROM users WHERE lower(username)=lower(?) OR lower(email)=lower(?)",
                      (username_or_email.strip(), username_or_email.strip())).fetchone()
    con.close()
    if not row:
        return None
    salt, digest = row[2], row[3]
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    if hmac.compare_digest(candidate, digest):
        return {"id": row[0], "username": row[1]}
    return None

def get_history(user_id):
    con = db()
    rows = con.execute("SELECT filename, prediction, confidence, created_at FROM predictions WHERE user_id=? ORDER BY id DESC",
                       (user_id,)).fetchall()
    con.close()
    return rows

def save_prediction(user_id, filename, prediction, confidence):
    con = db()
    con.execute("INSERT INTO predictions(user_id,filename,prediction,confidence,created_at) VALUES(?,?,?,?,?)",
                (user_id, filename, prediction, float(confidence), datetime.now().strftime("%Y-%m-%d %H:%M")))
    con.commit()
    con.close()

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

def get_label_names():
    # Keep this order aligned with the integer labels used when training the model.
    # If the notebook's label list was different, change this list to match it.
    return ["Fresh", "Rotten"]

def predict_image(image):
    model = load_model()
    resized = image.convert("RGB").resize((96, 96))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    labels = get_label_names()
    if idx >= len(labels):
        raise ValueError("The model output count does not match the label list.")
    return labels[idx], float(probs[idx]) * 100, probs

def logout():
    for key in ("user", "page", "last_result", "uploaded_image_bytes"):
        st.session_state.pop(key, None)
    st.rerun()

# ---------- Session state ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ---------- Login / signup ----------
if st.session_state.user is None:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("""
        <div style="padding:2.5rem .4rem 1rem;">
          <div class="eyebrow">SMART ANALYSIS · AI POWERED</div>
          <h1 style="font-size:3.2rem;line-height:1.08;margin-top:1rem;">Know your fruit.<br><span style="color:#25e5a4;">Choose fresher.</span></h1>
          <p class="subtle" style="font-size:1.1rem;max-width:520px;">A simple AI assistant that classifies fruit images as Fresh or Rotten using your trained machine-learning model.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="panel"><span class="eyebrow">YOUR WORKSPACE</span><h3>One image. One prediction.</h3><p class="subtle">Upload fruit photos, review confidence scores, and revisit your previous predictions.</p></div>', unsafe_allow_html=True)
        st.caption("Student machine-learning project · TensorFlow + Streamlit")
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
        with login_tab:
            st.markdown("### Welcome back")
            st.caption("Sign in to continue to your workspace.")
            with st.form("login_form"):
                login_id = st.text_input("Username or email")
                login_pw = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                user = authenticate(login_id, login_pw)
                if user:
                    st.session_state.user = user
                    st.session_state.page = "Dashboard"
                    st.rerun()
                else:
                    st.error("Sign-in failed. Check your username/email and password.")
        with signup_tab:
            st.markdown("### Create your account")
            st.caption("Use at least 8 characters for your password.")
            with st.form("signup_form"):
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_pw = st.text_input("Password", type="password")
                confirm_pw = st.text_input("Confirm password", type="password")
                signup_submit = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if signup_submit:
                if new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                else:
                    ok, message = register_user(new_username, new_email, new_pw)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------- App navigation ----------
user = st.session_state.user
with st.sidebar:
    st.markdown("## 🍎 Fruit Quality AI")
    st.caption("SMART FRUIT ANALYSIS")
    st.divider()
    pages = ["Dashboard", "Predict quality", "History", "Profile"]
    for page_name in pages:
        if st.button(("▣  " if st.session_state.page == page_name else "   ") + page_name,
                     key="nav_" + page_name, use_container_width=True,
                     type="primary" if st.session_state.page == page_name else "secondary"):
            st.session_state.page = page_name
            st.rerun()
    st.divider()
    st.markdown(f"**{user['username']}**")
    st.caption("Signed in")
    if st.button("Log out", use_container_width=True):
        logout()

# Load history for user-specific dashboard.
history = get_history(user["id"])
st.markdown(f'<div class="eyebrow">FRUIT QUALITY AI / {st.session_state.page.upper()}</div>', unsafe_allow_html=True)

if st.session_state.page == "Dashboard":
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">YOUR AI WORKSPACE</div>
      <h1>Welcome back, {user['username']} 👋</h1>
      <p class="subtle">Make fruit quality assessment simpler, one image at a time.</p>
    </div>""", unsafe_allow_html=True)
    total = len(history)
    fresh_count = sum(1 for row in history if row[1] == "Fresh")
    rotten_count = sum(1 for row in history if row[1] == "Rotten")
    avg_conf = np.mean([row[2] for row in history]) if history else 0
    metrics = [
        ("TOTAL PREDICTIONS", total, "All your saved results"),
        ("FRESH", fresh_count, "Predicted as fresh"),
        ("ROTTEN", rotten_count, "Predicted as rotten"),
        ("AVG. CONFIDENCE", f"{avg_conf:.1f}%" if history else "—", "Across saved predictions"),
    ]
    cols = st.columns(4)
    for col, (label, value, caption) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="subtle" style="font-size:.8rem;margin-top:6px;">{caption}</div></div>', unsafe_allow_html=True)
    st.markdown("### Quick actions")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="panel"><h3>🍏 Predict fruit quality</h3><p class="subtle">Upload an image and get a model prediction.</p></div>', unsafe_allow_html=True)
        if st.button("Start a prediction →", type="primary", use_container_width=True):
            st.session_state.page = "Predict quality"
            st.rerun()
    with c2:
        st.markdown('<div class="panel"><h3>🗂️ View prediction history</h3><p class="subtle">Review your previous results and confidence scores.</p></div>', unsafe_allow_html=True)
        if st.button("Open history →", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()
    st.info("Model predictions are estimates, not a guarantee of food safety.")

elif st.session_state.page == "Predict quality":
    st.markdown("## Predict fruit quality")
    st.markdown('<p class="subtle">Choose a clear image showing one fruit. Supported formats: JPG, JPEG, PNG.</p>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload fruit image", type=["jpg", "jpeg", "png"], key="fruit_upload")
    if upload:
        try:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption=upload.name, width=420)
            if st.button("Analyze image", type="primary"):
                with st.spinner("Analyzing with your trained model..."):
                    prediction, confidence, probabilities = predict_image(image)
                save_prediction(user["id"], upload.name, prediction, confidence)
                st.session_state.last_result = {
                    "filename": upload.name, "prediction": prediction, "confidence": confidence,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.success("Analysis complete.")
                st.rerun()
        except Exception as e:
            st.error(f"Could not read or analyze this image: {e}")
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        st.markdown("---")
        is_fresh = result["prediction"].lower() == "fresh"
        st.markdown(f'<div class="panel"><div class="eyebrow">LATEST RESULT</div><h2 style="color:{"#4bf0b0" if is_fresh else "#ff8c91"};">{"✓ Fresh" if is_fresh else "⚠ Rotten"}</h2><p class="subtle">Predicted class for <b>{result["filename"]}</b></p><h3>{result["confidence"]:.2f}% confidence</h3></div>', unsafe_allow_html=True)
        st.progress(min(max(result["confidence"] / 100, 0), 1))
        st.caption("Confidence is the model's score for its chosen class; it is not a guarantee of correctness or food safety.")

elif st.session_state.page == "History":
    st.markdown("## Prediction history")
    st.caption("Only predictions made while signed into your account are shown here.")
    if not history:
        st.markdown('<div class="panel"><h3>No predictions yet</h3><p class="subtle">Your future results will appear here.</p></div>', unsafe_allow_html=True)
        if st.button("Make your first prediction", type="primary"):
            st.session_state.page = "Predict quality"
            st.rerun()
    else:
        import pandas as pd
        df = pd.DataFrame(history, columns=["Image file", "Prediction", "Confidence (%)", "Date & time"])
        df["Confidence (%)"] = df["Confidence (%)"].map(lambda x: round(float(x), 2))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download history as CSV", df.to_csv(index=False).encode("utf-8"), "fruit_prediction_history.csv", "text/csv")

elif st.session_state.page == "Profile":
    st.markdown("## Profile")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Account details")
    st.write(f"**Username:** {user['username']}")
    st.caption("This local demo stores account details and prediction history in a SQLite database on this computer.")
    st.warning("This login is intended for a local student project, not deployment on the public internet without additional security work.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Fruit Quality AI · Student machine-learning project · TensorFlow + Streamlit")
