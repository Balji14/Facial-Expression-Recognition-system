"""
AI-Powered Facial Expression Recognition System
Premium UI — Final Year Project Showcase
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import json
import hashlib
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter
import io

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE — logic unchanged
# ═══════════════════════════════════════════════════════════════════════════════
_cfg = {}
if os.path.exists('model_config.json'):
    with open('model_config.json') as f:
        _cfg = json.load(f)

_preprocessing = _cfg.get('preprocessing', 'mobilenet_v2')
emotion_labels = _cfg.get('labels', ['anger', 'contempt', 'disgust', 'fear',
                                      'happy', 'neutral', 'sad', 'surprise'])

if _preprocessing == 'efficientnet':
    _preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
else:
    _preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SIZE = _cfg.get('img_size', 96)


@st.cache_resource
def load_model():
    if os.path.exists('emotion_model.keras'):
        m = tf.keras.models.load_model('emotion_model.keras', compile=False)
        return m, m.input_shape[1]
    return None, IMG_SIZE


model, IMG_SIZE = load_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
prediction_cache: dict = {}

EMOJI_MAP = {
    'anger': '😠', 'contempt': '😒', 'disgust': '🤢',
    'fear': '😨', 'happy': '😄', 'neutral': '😐',
    'sad': '😢', 'surprise': '😲',
}
EMOTION_COLORS = {
    'anger': '#EF4444', 'contempt': '#F97316', 'disgust': '#84CC16',
    'fear': '#A855F7', 'happy': '#EAB308', 'neutral': '#6B7280',
    'sad': '#3B82F6', 'surprise': '#EC4899',
}


def _hash_image(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()


def detect_faces_fast(gray):
    h, w = gray.shape[:2]
    scale = 1.0
    if w > 640:
        scale = 640.0 / w
        small = cv2.resize(gray, (640, int(h * scale)))
    else:
        small = gray
    faces = face_cascade.detectMultiScale(
        small, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return []
    return [
        (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
        for (x, y, fw, fh) in faces
    ]


def _predict_array(img_arr: np.ndarray) -> np.ndarray:
    if model is not None:
        return model.predict(img_arr, verbose=0)
    raise RuntimeError("No model available")


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _normalize_lighting(roi_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((_clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _predict_with_tta(roi_rgb: np.ndarray) -> np.ndarray:
    roi_rgb = _normalize_lighting(roi_rgb)

    def _prep(img):
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        arr = _preprocess_fn(np.asarray(resized, dtype=np.float32))
        return np.expand_dims(arr, axis=0)
    pred_orig    = _predict_array(_prep(roi_rgb))
    pred_flipped = _predict_array(_prep(np.fliplr(roi_rgb)))
    return (pred_orig + pred_flipped) / 2.0


def predict_from_frame(frame_rgb: np.ndarray):
    key = _hash_image(frame_rgb.tobytes())
    if key in prediction_cache:
        return prediction_cache[key]

    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = detect_faces_fast(gray)
    if len(faces) == 0:
        return frame_rgb, "No face detected in this image. Please upload or capture a clear photo with a visible face."

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    frame_h, frame_w = frame_rgb.shape[:2]
    pad = int(0.2 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame_w, x + w + pad), min(frame_h, y + h + pad)
    roi_rgb = frame_rgb[y0:y1, x0:x1]

    prediction = _predict_with_tta(roi_rgb)
    max_index  = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    emotion    = emotion_labels[max_index]
    probs      = {emotion_labels[i]: round(float(prediction[0][i]) * 100, 1)
                  for i in range(len(emotion_labels))}

    output = frame_rgb.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 230, 100), 3)
    cv2.putText(output, f"{emotion}  {confidence * 100:.1f}%",
                (x, max(y - 12, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 100), 2)

    result = {
        'emotion':    emotion,
        'confidence': round(confidence * 100, 1),
        'emoji':      EMOJI_MAP.get(emotion, ''),
        'probs':      probs,
        'timestamp':  datetime.now().strftime('%H:%M:%S'),
    }

    if len(prediction_cache) > 256:
        prediction_cache.pop(next(iter(prediction_cache)))
    prediction_cache[key] = (output, result)
    return output, result


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if 'dark_mode'   not in st.session_state: st.session_state.dark_mode   = False
if 'history'     not in st.session_state: st.session_state.history     = []
if 'last_result' not in st.session_state: st.session_state.last_result = None
if 'last_image'  not in st.session_state: st.session_state.last_image  = None

dark = st.session_state.dark_mode


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ═══════════════════════════════════════════════════════════════════════════════
def inject_styles(dark: bool):
    bg      = "#0F172A"  if dark else "#F8FAFC"
    text1   = "#F1F5F9"  if dark else "#1E293B"
    text2   = "#94A3B8"  if dark else "#64748B"
    glass   = "rgba(30,41,59,0.75)"    if dark else "rgba(255,255,255,0.75)"
    glass_b = "rgba(255,255,255,0.1)"  if dark else "rgba(255,255,255,0.7)"
    card    = "rgba(30,41,59,0.85)"    if dark else "rgba(255,255,255,0.9)"
    input_bg= "rgba(15,23,42,0.5)"     if dark else "rgba(255,255,255,0.6)"
    shadow  = "rgba(0,0,0,0.35)"       if dark else "rgba(0,0,0,0.07)"
    row_bg  = "rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.02)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Strip Streamlit chrome ──────────────────────────────────────────── */
    #MainMenu, footer, header {{ visibility: hidden !important; }}
    .stDeployButton, [data-testid="stToolbar"] {{ display: none !important; }}
    section[data-testid="stSidebar"] {{ display: none !important; }}
    .main .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}

    /* ── Base ────────────────────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: {bg} !important;
        color: {text1};
        scroll-behavior: smooth;
    }}
    ::-webkit-scrollbar {{ width: 7px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(#0D9488, #8B5CF6);
        border-radius: 4px;
    }}

    /* ── Animations ──────────────────────────────────────────────────────── */
    @keyframes fadeInDown {{
        from {{ opacity:0; transform:translateY(-16px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity:0; transform:translateY(16px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes bounceIn {{
        0%   {{ transform:scale(0.3); opacity:0; }}
        60%  {{ transform:scale(1.1); }}
        100% {{ transform:scale(1);   opacity:1; }}
    }}
    @keyframes gradientFlow {{
        0%   {{ background-position:0%   50%; }}
        50%  {{ background-position:100% 50%; }}
        100% {{ background-position:0%   50%; }}
    }}
    @keyframes pulseRing {{
        0%,100% {{ box-shadow:0 0 0 0   rgba(13,148,136,0.15); }}
        50%      {{ box-shadow:0 0 0 10px rgba(13,148,136,0); }}
    }}
    @keyframes floatBg {{
        0%,100% {{ transform:translate(0,0)   rotate(0deg); }}
        33%     {{ transform:translate(2%,1%) rotate(1deg); }}
        66%     {{ transform:translate(-1%,2%)rotate(-1deg); }}
    }}
    @keyframes slideInRight {{
        from {{ opacity:0; transform:translateX(20px); }}
        to   {{ opacity:1; transform:translateX(0); }}
    }}
    @keyframes spin {{
        from {{ transform:rotate(0deg); }}
        to   {{ transform:rotate(360deg); }}
    }}

    /* ── Navbar ──────────────────────────────────────────────────────────── */
    .navbar {{
        position: sticky; top: 0; z-index: 1000;
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 48px;
        background: {glass};
        backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        border-bottom: 1px solid {glass_b};
        box-shadow: 0 2px 24px {shadow};
    }}
    .nav-brand {{
        display: flex; align-items: center; gap: 10px;
        font-weight: 800; font-size: 1.05rem;
        background: linear-gradient(135deg, #0D9488, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .nav-pill {{
        background: linear-gradient(135deg, #0D9488, #8B5CF6);
        color: #fff; padding: 4px 12px; border-radius: 100px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
    }}
    .nav-links {{
        display: flex; align-items: center; gap: 6px;
    }}
    .nav-tag {{
        background: rgba(13,148,136,0.08);
        border: 1px solid rgba(13,148,136,0.18);
        color: #0D9488; padding: 5px 14px; border-radius: 8px;
        font-size: 0.82rem; font-weight: 600;
    }}

    /* ── Hero ────────────────────────────────────────────────────────────── */
    .hero {{
        position: relative; overflow: hidden;
        padding: 96px 48px 72px; text-align: center;
        background: {'linear-gradient(135deg,#0F172A 0%,#1A2744 50%,#0F172A 100%)' if dark
                     else 'linear-gradient(135deg,#F0FDFA 0%,#EDE9FE 60%,#F0FDFA 100%)'};
    }}
    .hero::before {{
        content:''; position:absolute; inset:-50%;
        background: radial-gradient(circle at 28% 32%, rgba(13,148,136,.18) 0%, transparent 52%),
                    radial-gradient(circle at 72% 68%, rgba(139,92,246,.18) 0%, transparent 52%);
        animation: floatBg 10s ease-in-out infinite;
        pointer-events: none;
    }}
    .hero-badge {{
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(13,148,136,.1); border: 1px solid rgba(13,148,136,.3);
        color: #0D9488; padding: 8px 22px; border-radius: 100px;
        font-size: .83rem; font-weight: 700; letter-spacing: .4px;
        margin-bottom: 28px;
        animation: fadeInDown .6s ease both;
    }}
    .hero-title {{
        font-size: clamp(1.9rem, 5.5vw, 3.6rem);
        font-weight: 900; line-height: 1.14; margin-bottom: 20px;
        background: {'linear-gradient(135deg,#5EEAD4,#A78BFA,#5EEAD4)' if dark
                     else 'linear-gradient(135deg,#0F766E,#6D28D9,#0D9488)'};
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradientFlow 5s ease infinite, fadeInDown .65s ease both;
    }}
    .hero-sub {{
        font-size: clamp(.9rem, 2vw, 1.12rem);
        color: {text2}; max-width: 660px; margin: 0 auto 44px;
        line-height: 1.75; animation: fadeInUp .7s ease both;
    }}
    .hero-stats {{
        display: flex; justify-content: center; flex-wrap: wrap; gap: 36px;
        animation: fadeInUp .8s ease both;
    }}
    .hero-stat-val {{
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #0D9488, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .hero-stat-lbl {{
        font-size: .78rem; color: {text2}; font-weight: 500; margin-top: 2px;
        text-transform: uppercase; letter-spacing: .5px;
    }}

    /* ── Section layout ──────────────────────────────────────────────────── */
    .sec {{
        max-width: 1160px; margin: 0 auto; padding: 60px 40px;
    }}
    .sec-badge {{
        display: inline-block;
        background: linear-gradient(135deg, rgba(13,148,136,.08), rgba(139,92,246,.08));
        border: 1px solid rgba(13,148,136,.2);
        color: #0D9488; padding: 5px 15px; border-radius: 100px;
        font-size: .75rem; font-weight: 700; letter-spacing: 1px;
        text-transform: uppercase; margin-bottom: 10px;
    }}
    .sec-title {{
        font-size: clamp(1.3rem, 3vw, 1.9rem);
        font-weight: 800; color: {text1}; margin-bottom: 6px;
    }}
    .sec-desc {{ color: {text2}; font-size: .93rem; line-height: 1.65; }}
    .divider {{
        height: 1px; margin: 6px 0 28px;
        background: linear-gradient(to right, transparent,
            rgba(13,148,136,.35), rgba(139,92,246,.35), transparent);
    }}

    /* ── Glass card ──────────────────────────────────────────────────────── */
    .card {{
        background: {card};
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border: 1px solid {glass_b}; border-radius: 20px;
        box-shadow: 0 4px 28px {shadow}; padding: 28px;
        transition: transform .3s ease, box-shadow .3s ease;
        animation: fadeInUp .5s ease both;
    }}
    .card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 14px 44px {shadow};
    }}

    /* ── Stat cards ──────────────────────────────────────────────────────── */
    .stat-card {{
        background: {card}; border: 1px solid {glass_b};
        border-radius: 16px; padding: 22px 16px; text-align: center;
        position: relative; overflow: hidden;
        transition: transform .25s ease;
    }}
    .stat-card::before {{
        content:''; position:absolute; top:0; left:0; right:0; height:3px;
        background: linear-gradient(90deg, #0D9488, #8B5CF6);
    }}
    .stat-card:hover {{ transform: translateY(-3px); }}
    .stat-val {{
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(135deg, #0D9488, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .stat-lbl {{
        font-size: .75rem; color: {text2}; font-weight: 600;
        text-transform: uppercase; letter-spacing: .6px; margin-top: 4px;
    }}

    /* ── Result card ─────────────────────────────────────────────────────── */
    .result-card {{
        background: linear-gradient(135deg,
            rgba(13,148,136,.08), rgba(139,92,246,.08));
        border: 1px solid rgba(13,148,136,.25);
        border-radius: 20px; padding: 32px 24px; text-align: center;
        animation: pulseRing 2.5s ease infinite;
    }}
    .result-emoji {{
        font-size: 4.5rem; display: block; margin-bottom: 10px;
        animation: bounceIn .6s cubic-bezier(.36,.07,.19,.97) both;
    }}
    .result-emotion {{
        font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;
        color: {text1};
    }}
    .result-sub {{ font-size: .88rem; color: {text2}; margin-bottom: 16px; }}
    .conf-bar {{
        width: 100%; height: 10px;
        background: rgba({'255,255,255' if dark else '0,0,0'},.1);
        border-radius: 5px; overflow: hidden;
    }}
    .conf-fill {{
        height: 100%;
        background: linear-gradient(90deg, #0D9488, #8B5CF6);
        border-radius: 5px; transition: width 1.2s cubic-bezier(.4,0,.2,1);
    }}

    /* ── History rows ────────────────────────────────────────────────────── */
    .h-row {{
        display: flex; align-items: center; gap: 14px;
        padding: 11px 16px; border-radius: 12px;
        background: {row_bg}; border: 1px solid {glass_b};
        margin-bottom: 8px; transition: background .2s ease;
        animation: slideInRight .3s ease both;
    }}
    .h-row:hover {{ background: rgba(13,148,136,.06); border-color: rgba(13,148,136,.2); }}
    .h-dot {{
        width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    }}
    .h-emo {{ font-weight: 600; font-size: .9rem; color: {text1}; flex: 1; }}
    .h-conf {{ font-size: .82rem; color: {text2}; }}
    .h-time {{ font-size: .75rem; color: {text2}; }}

    /* ── Workflow ────────────────────────────────────────────────────────── */
    .wf-wrap {{
        display: flex; align-items: center; justify-content: center;
        flex-wrap: wrap; gap: 0; padding: 16px 0;
    }}
    .wf-step {{
        background: {card}; border: 1px solid {glass_b};
        border-radius: 16px; padding: 20px 16px; text-align: center;
        min-width: 110px; max-width: 130px; position: relative;
        transition: all .3s ease;
    }}
    .wf-step:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(13,148,136,.2);
        border-color: rgba(13,148,136,.4);
    }}
    .wf-num {{
        position: absolute; top: -10px; left: 50%; transform: translateX(-50%);
        background: linear-gradient(135deg, #0D9488, #8B5CF6);
        color: #fff; width: 22px; height: 22px; border-radius: 50%;
        font-size: .68rem; font-weight: 800;
        display: flex; align-items: center; justify-content: center;
    }}
    .wf-icon {{ font-size: 1.9rem; margin-bottom: 8px; }}
    .wf-lbl {{ font-size: .75rem; font-weight: 600; color: {text1}; line-height: 1.3; }}
    .wf-arr {{ color: #0D9488; font-size: 1.6rem; padding: 0 6px; flex-shrink: 0; }}

    /* ── Tech grid ───────────────────────────────────────────────────────── */
    .tech-grid {{
        display: grid; grid-template-columns: repeat(auto-fit,minmax(185px,1fr));
        gap: 14px; margin-top: 18px;
    }}
    .tech-item {{
        background: {row_bg}; border: 1px solid {glass_b};
        border-radius: 12px; padding: 16px;
    }}
    .tech-lbl {{
        font-size: .72rem; color: {text2}; text-transform: uppercase;
        letter-spacing: .6px; font-weight: 600; margin-bottom: 4px;
    }}
    .tech-val {{ font-size: .92rem; font-weight: 700; color: {text1}; }}

    /* ── Footer ──────────────────────────────────────────────────────────── */
    .footer {{
        background: {'#080E1E' if dark else '#1E293B'};
        padding: 52px 40px; text-align: center; margin-top: 80px;
    }}
    .footer-brand {{
        font-size: 1.2rem; font-weight: 800; margin-bottom: 12px;
        background: linear-gradient(135deg, #0D9488, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .footer-text {{
        font-size: .84rem; color: #94A3B8; line-height: 1.7;
        max-width: 520px; margin: 0 auto 20px;
    }}
    .footer-chips {{
        display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;
    }}
    .footer-chip {{
        background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
        color: #94A3B8; padding: 5px 14px; border-radius: 100px;
        font-size: .76rem; font-weight: 500;
    }}
    .footer-copy {{ font-size: .76rem; color: #475569; margin-top: 24px; }}

    /* ── Streamlit widget overrides ──────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, #0D9488, #8B5CF6) !important;
        color: #fff !important; border: none !important;
        border-radius: 12px !important; padding: 12px 28px !important;
        font-weight: 600 !important; font-size: .93rem !important;
        box-shadow: 0 4px 16px rgba(13,148,136,.3) !important;
        transition: all .3s ease !important; width: 100%;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(13,148,136,.4) !important;
    }}
    [data-testid="stFileUploadDropzone"] {{
        border: 2px dashed rgba(13,148,136,.4) !important;
        border-radius: 16px !important; background: transparent !important;
        transition: all .3s ease !important;
    }}
    [data-testid="stFileUploadDropzone"]:hover {{
        border-color: #0D9488 !important;
        background: rgba(13,148,136,.04) !important;
    }}
    [data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] * {{
        color: {text1} !important;
    }}
    [data-testid="stFileUploaderFile"] small {{
        color: {text2} !important;
    }}
    [data-testid="stCameraInput"] > div > div:first-child {{
        border-radius: 16px !important; overflow: hidden !important;
        border: 2px solid rgba(13,148,136,.2) !important;
    }}
    [data-testid="stExpander"] {{
        background: {card} !important;
        border: 1px solid {glass_b} !important; border-radius: 16px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {row_bg} !important;
        border-radius: 12px !important; padding: 4px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px !important; font-weight: 500 !important;
        color: {text2} !important; transition: all .2s ease !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #0D9488, #8B5CF6) !important;
        color: #fff !important;
    }}
    .stSpinner > div {{ border-top-color: #0D9488 !important; }}
    div[data-testid="stMarkdownContainer"] p {{ color: {text1}; }}

    /* ── Responsive ──────────────────────────────────────────────────────── */
    @media (max-width: 768px) {{
        .navbar {{ padding: 12px 18px; }}
        .hero   {{ padding: 60px 18px 44px; }}
        .sec    {{ padding: 40px 18px; }}
        .hero-stats {{ gap: 22px; }}
        .wf-wrap {{ flex-direction: column; }}
        .wf-arr  {{ transform: rotate(90deg); padding: 4px 0; }}
        .nav-links .nav-tag:not(:last-child) {{ display: none; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _chart_layout(fig, h, dark):
    fig.update_layout(
        height=h, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#F1F5F9' if dark else '#1E293B',
    )
    return fig


def gauge_chart(confidence: float, emotion: str, dark: bool):
    color = EMOTION_COLORS.get(emotion, '#0D9488')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={'suffix': '%', 'font': {'size': 38, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1,
                     'tickcolor': '#94A3B8', 'tickfont': {'size': 10}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40],  'color': 'rgba(239,68,68,.08)'},
                {'range': [40, 70], 'color': 'rgba(234,179,8,.08)'},
                {'range': [70, 100],'color': 'rgba(13,148,136,.08)'},
            ],
        }
    ))
    return _chart_layout(fig, 210, dark)


def probs_chart(probs: dict, dark: bool):
    emotions = list(probs.keys())
    values   = list(probs.values())
    colors   = [EMOTION_COLORS.get(e, '#6B7280') for e in emotions]
    fig = go.Figure(go.Bar(
        y=emotions, x=values, orientation='h',
        marker=dict(color=colors, line_width=0, cornerradius=6),
        text=[f"{v}%" for v in values],
        textposition='inside', insidetextanchor='start',
        textfont={'size': 11, 'color': '#fff'},
    ))
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 105]),
        yaxis=dict(showgrid=False,
                   tickfont={'size': 12, 'color': '#F1F5F9' if dark else '#1E293B'}),
        showlegend=False,
    )
    return _chart_layout(fig, 290, dark)


def history_donut(history: list, dark: bool):
    counts   = Counter(h['emotion'] for h in history)
    emotions = list(counts.keys())
    values   = list(counts.values())
    colors   = [EMOTION_COLORS.get(e, '#6B7280') for e in emotions]
    labels   = [f"{EMOJI_MAP.get(e,'')} {e.capitalize()}" for e in emotions]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=colors, line=dict(width=0)),
        textinfo='percent', hoverinfo='label+value+percent',
        textfont={'size': 12},
    ))
    fig.update_layout(
        legend=dict(font={'size': 11}, orientation='v'),
        annotations=[dict(text=f"<b>{sum(values)}</b><br><span style='font-size:10px'>scans</span>",
                          x=0.5, y=0.5, font_size=18, showarrow=False,
                          font_color='#F1F5F9' if dark else '#1E293B')],
    )
    return _chart_layout(fig, 280, dark)


# ═══════════════════════════════════════════════════════════════════════════════
# UI — RENDER
# ═══════════════════════════════════════════════════════════════════════════════
inject_styles(dark)

# ── Dark mode toggle (top-right) ──────────────────────────────────────────────
_, _tcol = st.columns([22, 2])
with _tcol:
    icon = "☀️ Light" if dark else "🌙 Dark"
    if st.button(icon, key="theme_toggle"):
        st.session_state.dark_mode = not dark
        st.rerun()

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="nav-brand">
    😊 <span>EmotionAI</span>
    <span class="nav-pill">FYP 2025</span>
  </div>
  <div class="nav-links">
    <span class="nav-tag">🧠 Deep Learning</span>
    <span class="nav-tag">👁️ Computer Vision</span>
    <span class="nav-tag">⚡ Real-time</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🤖 Final Year Project &nbsp;·&nbsp; Deep Learning &nbsp;·&nbsp; Computer Vision</div>
  <h1 class="hero-title">AI-Powered Facial Expression<br>Recognition System</h1>
  <p class="hero-sub">
    A deep learning system using MobileNetV2 transfer learning and OpenCV face detection
    to classify 8 human emotions in real-time with Test-Time Augmentation for improved accuracy.
  </p>
  <div class="hero-stats">
    <div><div class="hero-stat-val">8</div><div class="hero-stat-lbl">Emotion Classes</div></div>
    <div><div class="hero-stat-val">MobileNetV2</div><div class="hero-stat-lbl">Architecture</div></div>
    <div><div class="hero-stat-val">TTA</div><div class="hero-stat-lbl">Augmentation</div></div>
    <div><div class="hero-stat-val">Real-time</div><div class="hero-stat-lbl">Detection</div></div>
    <div><div class="hero-stat-val">LRU</div><div class="hero-stat-lbl">Prediction Cache</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Detection section ─────────────────────────────────────────────────────────
st.markdown("""
<div class="sec">
  <div class="sec-badge">🔬 Detection Engine</div>
  <div class="sec-title">Analyze Your Facial Expression</div>
  <div class="sec-desc">Capture a live photo with your webcam or upload an image to receive
      an instant emotion prediction powered by deep learning.</div>
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tab_cam, tab_up = st.tabs(["📷  Webcam Capture", "📁  Upload Image"])

        with tab_cam:
            st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
            cam_img = st.camera_input("", label_visibility="collapsed")

        with tab_up:
            st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
            up_img = st.file_uploader(
                "Drag & drop or click to browse",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible",
            )

        img_file = cam_img or up_img

        if img_file is not None:
            if st.button("🚀  Detect Emotion", key="detect"):
                raw       = np.frombuffer(img_file.getvalue(), np.uint8)
                frame_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                with st.spinner("Analyzing expression..."):
                    annotated, result = predict_from_frame(frame_rgb)
                if isinstance(result, str):
                    st.error(f"🚫 {result}")
                    st.session_state.last_result = None
                    st.session_state.last_image  = None
                else:
                    st.session_state.last_result = result
                    st.session_state.last_image  = annotated
                    st.session_state.history.insert(0, result)
                    if len(st.session_state.history) > 20:
                        st.session_state.history = st.session_state.history[:20]
                    st.toast(f"{result['emoji']} {result['emotion'].capitalize()} detected!", icon="✅")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if st.session_state.last_image is not None:
            res = st.session_state.last_result

            if res:
                color = EMOTION_COLORS.get(res['emotion'], '#0D9488')
                st.markdown(f"""
                <div class="result-card">
                  <span class="result-emoji">{res['emoji']}</span>
                  <div class="result-emotion" style="color:{color}">
                    {res['emotion'].capitalize()}
                  </div>
                  <div class="result-sub">Confidence Score</div>
                  <div class="conf-bar">
                    <div class="conf-fill" style="width:{res['confidence']}%"></div>
                  </div>
                  <div style="font-size:.85rem;margin-top:8px;font-weight:700;color:{color}">
                    {res['confidence']}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(gauge_chart(res['confidence'], res['emotion'], dark),
                                use_container_width=True, config={'displayModeBar': False})

                # Download annotated image
                img_bgr = cv2.cvtColor(st.session_state.last_image, cv2.COLOR_RGB2BGR)
                _, buf   = cv2.imencode('.jpg', img_bgr)
                st.download_button(
                    "⬇️  Download Result", data=buf.tobytes(),
                    file_name=f"emotion_{res['emotion']}.jpg",
                    mime="image/jpeg", key="dl",
                )
            else:
                st.image(st.session_state.last_image, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:60px 24px;min-height:320px;
                 display:flex;flex-direction:column;align-items:center;justify-content:center;">
              <div style="font-size:3.5rem;margin-bottom:16px">🎭</div>
              <div style="font-size:1rem;font-weight:600;margin-bottom:8px">
                Your result will appear here
              </div>
              <div style="font-size:.85rem;color:#94A3B8">
                Capture or upload a photo and click Detect Emotion
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── Probability distribution ──────────────────────────────────────────────────
if st.session_state.last_result:
    res = st.session_state.last_result
    st.markdown("""
    <div class="sec" style="padding-top:20px">
      <div class="sec-badge">📊 Probability Distribution</div>
      <div class="sec-title">All Emotion Probabilities</div>
      <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        _, c, _ = st.columns([0.5, 9, 0.5])
        with c:
            st.plotly_chart(probs_chart(res['probs'], dark),
                            use_container_width=True, config={'displayModeBar': False})

# ── Dashboard ─────────────────────────────────────────────────────────────────
if st.session_state.history:
    h = st.session_state.history
    total      = len(h)
    avg_conf   = round(sum(x['confidence'] for x in h) / total, 1)
    top_emotion= Counter(x['emotion'] for x in h).most_common(1)[0][0]
    top_emoji  = EMOJI_MAP.get(top_emotion, '')

    st.markdown("""
    <div class="sec" style="padding-top:10px">
      <div class="sec-badge">📈 AI Dashboard</div>
      <div class="sec-title">Session Analytics</div>
      <div class="sec-desc">Live statistics from your current session predictions.</div>
      <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, val, lbl in [
        (sc1, str(total),               "Total Scans"),
        (sc2, f"{avg_conf}%",           "Avg Confidence"),
        (sc3, top_emotion.capitalize(), "Top Emotion"),
        (sc4, top_emoji,                "Emoji"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{val}</div>
              <div class="stat-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    dash_left, dash_right = st.columns([1, 1], gap="large")

    with dash_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🕑 Detection History**")
        for entry in h[:8]:
            col = EMOTION_COLORS.get(entry['emotion'], '#6B7280')
            st.markdown(f"""
            <div class="h-row">
              <div class="h-dot" style="background:{col}"></div>
              <div class="h-emo">{entry['emoji']} {entry['emotion'].capitalize()}</div>
              <div class="h-conf">{entry['confidence']}%</div>
              <div class="h-time">{entry['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑️ Clear History", key="clear"):
            st.session_state.history     = []
            st.session_state.last_result = None
            st.session_state.last_image  = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with dash_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🍩 Emotion Distribution**")
        st.plotly_chart(history_donut(h, dark),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

# ── Workflow pipeline ─────────────────────────────────────────────────────────
st.markdown("""
<div class="sec">
  <div class="sec-badge">⚙️ System Pipeline</div>
  <div class="sec-title">How It Works</div>
  <div class="sec-desc">End-to-end workflow from image capture to emotion classification.</div>
  <div class="divider"></div>
  <div class="wf-wrap">
    <div class="wf-step"><div class="wf-num">1</div>
      <div class="wf-icon">📸</div><div class="wf-lbl">Image Input</div></div>
    <div class="wf-arr">→</div>
    <div class="wf-step"><div class="wf-num">2</div>
      <div class="wf-icon">🔍</div><div class="wf-lbl">Face Detection</div></div>
    <div class="wf-arr">→</div>
    <div class="wf-step"><div class="wf-num">3</div>
      <div class="wf-icon">🔄</div><div class="wf-lbl">Preprocessing</div></div>
    <div class="wf-arr">→</div>
    <div class="wf-step"><div class="wf-num">4</div>
      <div class="wf-icon">🧠</div><div class="wf-lbl">Deep Learning Model</div></div>
    <div class="wf-arr">→</div>
    <div class="wf-step"><div class="wf-num">5</div>
      <div class="wf-icon">🔀</div><div class="wf-lbl">TTA Inference</div></div>
    <div class="wf-arr">→</div>
    <div class="wf-step"><div class="wf-num">6</div>
      <div class="wf-icon">🎭</div><div class="wf-lbl">Emotion Output</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Technical information ─────────────────────────────────────────────────────
st.markdown("""
<div class="sec" style="padding-top:0">
  <div class="sec-badge">🔬 Technical Details</div>
  <div class="sec-title">Model & System Information</div>
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    _, tc, _ = st.columns([0.5, 9, 0.5])
    with tc:
        with st.expander("📐 Model Architecture & Training", expanded=False):
            st.markdown("""
            <div class="tech-grid">
              <div class="tech-item"><div class="tech-lbl">Base Model</div>
                <div class="tech-val">MobileNetV2</div></div>
              <div class="tech-item"><div class="tech-lbl">Input Size</div>
                <div class="tech-val">96 × 96 × 3 RGB</div></div>
              <div class="tech-item"><div class="tech-lbl">Output Classes</div>
                <div class="tech-val">8 Emotions (Softmax)</div></div>
              <div class="tech-item"><div class="tech-lbl">Loss Function</div>
                <div class="tech-val">Categorical Cross-Entropy</div></div>
              <div class="tech-item"><div class="tech-lbl">Optimizer</div>
                <div class="tech-val">Adam (lr=0.0001)</div></div>
              <div class="tech-item"><div class="tech-lbl">Regularization</div>
                <div class="tech-val">L2 + Dropout (0.5)</div></div>
              <div class="tech-item"><div class="tech-lbl">Class Balancing</div>
                <div class="tech-val">Computed Class Weights</div></div>
              <div class="tech-item"><div class="tech-lbl">Inference</div>
                <div class="tech-val">2-way TTA (flip)</div></div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📚 Dataset Information", expanded=False):
            st.markdown("""
            <div class="tech-grid">
              <div class="tech-item"><div class="tech-lbl">Dataset</div>
                <div class="tech-val">FER (Facial Expression Recognition)</div></div>
              <div class="tech-item"><div class="tech-lbl">Training Images</div>
                <div class="tech-val">16,108 samples</div></div>
              <div class="tech-item"><div class="tech-lbl">Validation Images</div>
                <div class="tech-val">14,518 samples</div></div>
              <div class="tech-item"><div class="tech-lbl">Emotion Classes</div>
                <div class="tech-val">Anger, Contempt, Disgust, Fear,
                  Happy, Neutral, Sad, Surprise</div></div>
              <div class="tech-item"><div class="tech-lbl">Augmentation</div>
                <div class="tech-val">Rotation, Shift, Flip, Zoom</div></div>
              <div class="tech-item"><div class="tech-lbl">Preprocessing</div>
                <div class="tech-val">MobileNetV2 normalization</div></div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🛠️ Technology Stack", expanded=False):
            st.markdown("""
            <div class="tech-grid">
              <div class="tech-item"><div class="tech-lbl">Framework</div>
                <div class="tech-val">TensorFlow 2.x / Keras</div></div>
              <div class="tech-item"><div class="tech-lbl">Web App</div>
                <div class="tech-val">Streamlit</div></div>
              <div class="tech-item"><div class="tech-lbl">Face Detection</div>
                <div class="tech-val">OpenCV Haar Cascade</div></div>
              <div class="tech-item"><div class="tech-lbl">Visualization</div>
                <div class="tech-val">Plotly</div></div>
              <div class="tech-item"><div class="tech-lbl">Deployment</div>
                <div class="tech-val">Hugging Face Spaces</div></div>
              <div class="tech-item"><div class="tech-lbl">Language</div>
                <div class="tech-val">Python 3.11</div></div>
            </div>
            """, unsafe_allow_html=True)

# ── Emotion guide ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec" style="padding-top:0">
  <div class="sec-badge">🎭 Emotion Reference</div>
  <div class="sec-title">Detectable Emotions</div>
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    _, ec, _ = st.columns([0.5, 9, 0.5])
    with ec:
        cols = st.columns(4)
        for i, (emo, col_hex) in enumerate(EMOTION_COLORS.items()):
            with cols[i % 4]:
                emoji = EMOJI_MAP.get(emo, '')
                st.markdown(f"""
                <div class="stat-card" style="margin-bottom:12px">
                  <div style="font-size:2rem;margin-bottom:6px">{emoji}</div>
                  <div style="font-weight:700;color:{col_hex};font-size:.95rem">
                    {emo.capitalize()}
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-brand">😊 EmotionAI — Facial Expression Recognition</div>
  <p class="footer-text">
    Final Year Project demonstrating real-time facial emotion recognition using
    MobileNetV2 deep learning, OpenCV face detection, and Streamlit deployment
    on Hugging Face Spaces.
  </p>
  <div class="footer-chips">
    <span class="footer-chip">🧠 MobileNetV2</span>
    <span class="footer-chip">👁️ OpenCV</span>
    <span class="footer-chip">⚡ TensorFlow</span>
    <span class="footer-chip">🚀 Streamlit</span>
    <span class="footer-chip">🤗 HuggingFace</span>
    <span class="footer-chip">🎓 Final Year Project</span>
  </div>
  <div class="footer-copy">Built with ❤️ · AI-Powered Computer Vision · 2025</div>
</div>
""", unsafe_allow_html=True)
