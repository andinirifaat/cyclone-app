from networkx import center
import streamlit as st
import time
from datetime import date, datetime
from inference_validd import load_model, run_inference
from LLM_intepretation import interpret_boxes
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_lottie import st_lottie
import json
import streamlit.components.v1 as components
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_lottie(path):
    full_path = os.path.join(BASE_DIR, path)
    with open(full_path, "r") as f:
        return json.load(f)

st.set_page_config(
    page_title="AI-Based Tropical Cyclone Detection System",
    layout="wide",
)

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

if "modal" not in st.session_state:
    st.session_state.modal = None

if "component_value" not in st.session_state:
    st.session_state.component_value = None

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter';
}
.stApp {
    background: linear-gradient(135deg, #F0F9FF, #E0F2FE, #DBEAFE);
}
            
/* ===== GLOBAL STYLES ===== */

/* GLOBAL FONT SCALE */
html, body, [class*="css"]  {
    font-size: 20px !important;
}

/* TITLE CARD */
.card-title {
    font-size: 18px !important;
    font-weight: 700;
}

/* DESC */
.card-desc {
    font-size: 14px !important;
}

/* STEP NUMBER */
.card-step {
    font-size: 13px !important;
}

/* ===== WRAPPER CARD (INI TRICKNYA) ===== */
.input-card-wrapper {
    background: #FFFFFF;
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #E3F2FD;
    box-shadow: 0 8px 24px rgba(26,107,181,0.12);
}

/* label */
.input-label {
    font-weight: 600;
    color: #0F2A44;
    margin-bottom: 8px;
}

/* ===== FIX HITAM (PENTING BANGET) ===== */
.stFileUploader,
.stFileUploader * {
    background-color: #FFFFFF !important;
}

.stDateInput,
.stDateInput * {
    background-color: #FFFFFF !important;
}

/* border */
.stFileUploader > div {
    border: 2px dashed #C7DCEF !important;
    border-radius: 14px !important;
}

.stDateInput > div {
    border: 1px solid #C7DCEF !important;
    border-radius: 12px !important;
}


/* ===== TEXT DI FILE UPLOADER ===== */
.stFileUploader div[data-testid="stFileUploaderDropzone"] * {
    color: #0F2A44 !important;
}

/* teks kecil (limit file) */
.stFileUploader small {
    color: #0F2A44 !important;
}

/* fallback semua teks */
.stFileUploader span {
    color: #0F2A44 !important;
}
            
/* ===== TEXT DALAM DATE INPUT ===== */
.stDateInput input {
    color: #0F2A44 !important;
    font-weight: 500;
}

/* placeholder (kalau belum isi) */
.stDateInput input::placeholder {
    color: #94A3B8 !important;
}
            
/* Hero */
.hero-section {
    background: linear-gradient(160deg, #E0F2FE, #BAE6FD, #7DD3FC);
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 30% 50%, rgba(56,189,248,0.25) 0%, transparent 60%);
    animation: pulse-bg 8s ease-in-out infinite alternate;
}
@keyframes pulse-bg {
    to { background: radial-gradient(circle at 70% 50%, rgba(56,189,248,0.25) 0%, transparent 60%); }
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.85); border-radius: 999px;
    padding: 6px 16px; font-size: 0.85rem; font-weight: 600;
    color: #0369A1; margin-bottom: 1rem; backdrop-filter: blur(4px);
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #0C4A6E;
    line-height: 1.15; margin-bottom: 1rem; position: relative;
}
.hero-title span { color: #0284C7; }
.hero-sub {
    font-size: 1.05rem; color: #475569; max-width: 640px;
    margin: 0 auto; line-height: 1.7; position: relative;
}

/* Cards */
.info-card {
    background: #fff; border: 1px solid #BAE6FD; border-radius: 16px;
    padding: 1.5rem; box-shadow: 0 4px 24px -4px rgba(56,189,248,0.12);
    transition: box-shadow 0.3s;
    height: 100%;
}
.info-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(56,189,248,0.25);
}
.info-card {
    transition: all 0.25s ease;
}
.info-card .icon-wrap {
    width: 48px;
    height: 48px;
    border-radius: 12px;

    /* tetap terang */
    background: #E0F2FE;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 1.4rem;
    margin-bottom: 1rem;

    color: #0F2A44; /* fallback */
    text-shadow: 0 1px 2px rgba(0,0,0,0.25);
    filter: contrast(1.2) saturate(1.2);
}
.info-card h3 { font-size: 1.1rem; font-weight: 700; color: #0C4A6E; margin-bottom: 0.4rem; }
.info-card p { font-size: 0.88rem; color: #64748B; line-height: 1.6; }

.step-row {
    display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1.2rem;
}
.step-num {
    width: 40px; height: 40px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, #0284C7, #38BDF8);
    color: #fff; font-weight: 700; font-size: 0.9rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px rgba(2,132,199,0.35);
}
.step-text h4 { font-weight: 600; color: #0C4A6E; margin: 0; }
.step-text p { font-size: 0.88rem; color: #64748B; margin: 4px 0 0; }

.section-title {
    font-size: 1.6rem; font-weight: 700; color: #0F2A44 !important; 
    text-align: center; margin-bottom: 1.5rem;
}

/* TAB CONTAINER */
div[data-testid="stTabs"] {
    margin-top: 10px;
}

/* TAB BASE */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-weight: 600;
    border-radius: 10px;
    padding: 6px 14px;
    transition: all 0.2s ease;
}

/* TAB ACTIVE (ZOOM EARTH STYLE) */
button[data-baseweb="tab"][aria-selected="true"] {
    background: #E0F2FE !important;   /* biru muda */
    color: #0284C7 !important;
    border: none !important;
}

/* REMOVE UNDERLINE DEFAULT */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: none !important;
}

/* TAB AKTIF (ZOOM EARTH STYLE) */
button[data-baseweb="tab"][aria-selected="true"] {
    background: #E0F2FE !important;   /*biru muda */
    color: #0284C7 !important;
    border: none !important;
}        
.white-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 20px 22px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 8px 24px rgba(15, 42, 68, 0.06);
    margin-top: 16px;
}

.card-title {
    font-weight: 700;
    font-size: 18px;
    color: #0F2A44;
    margin-bottom: 12px;
}

.grok-box {
    background: #F8FAFC;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #E2E8F0;
    font-size: 14px;
    color: #1E293B;
    line-height: 1.6;
}

/* ===== FIX HEADING DI WHITE CARD ===== */
.white-card h1,
.white-card h2,
.white-card h3 {
    color: #0F2A44 !important;  /* dark navy */
}
            
.result-header {
    background: #fff; border-bottom: 1px solid #BAE6FD;
    padding: 1rem 1.5rem; border-radius: 16px 16px 0 0;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem;
}
.result-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #E0F2FE; color: #0284C7; border-radius: 999px;
    padding: 4px 12px; font-size: 0.78rem; font-weight: 600;
}
.result-badge::before {
    content: ''; width: 8px; height: 8px; border-radius: 50%;
    background: #0284C7; animation: blink 1.5s infinite;
}
@keyframes blink { 50% { opacity: 0.3; } }

.grok-box {
    background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 12px;
    padding: 1.2rem; font-size: 0.9rem; color: #334155; line-height: 1.7;
}
.grok-box strong { color: #0284C7; }

/* Streamlit overrides */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-weight: 600; color: #64748B;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0284C7 !important; border-bottom-color: #0284C7 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0284C7, #38BDF8) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    padding: 0.75rem 2rem !important; font-weight: 700 !important;
    font-size: 1.05rem !important; box-shadow: 0 4px 16px rgba(2,132,199,0.35) !important;
    width: auto !important;min-width: 240px !important;
}
div.stButton {
    position: relative !important;
    margin-bottom: 10px !important;
}
.stButton > button:hover { opacity: 0.9; }
.stButton > button:disabled { opacity: 0.4 !important; }
.card-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F2A44;
    margin-bottom: 0.8rem;
}
            
/* semua teks global */
body, .stApp {
    color: #0F2A44 !important;
}

/* slider label */
.stSlider label {
    color: #0F2A44 !important;
}

/* angka slider */
.stSlider span {
    color: #0F2A44 !important;
}

/* caption */
.stCaption {
    color: #0F2A44 !important;
    opacity: 1 !important;
}

/* tab text */
button[data-baseweb="tab"] {
    color: #334155 !important;
}

button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    height: 0px !important;
    padding: 0 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0284C7, #38BDF8) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    padding: 0.75rem 2rem !important; font-weight: 700 !important;
    font-size: 1.05rem !important; box-shadow: 0 4px 16px rgba(2,132,199,0.35) !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9; }
.stButton > button:disabled { opacity: 0.4 !important; }
            
footer { visibility: hidden; }
/* kasih jarak dalam card */
.white-card div[data-testid="stTabs"] {
    margin-top: 10px;
}

/* tab jadi inner */
button[data-baseweb="tab"] {
    border-radius: 10px !important;
    margin-right: 6px;
}
/* background tab area transparan */
div[data-testid="stTabs"] {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* parent card */
.element-container {
    position: relative;
}
/* ===== POPOVER CLEAN BLUE WHITE ===== */
div[data-testid="stPopover"] {
    background: linear-gradient(180deg, #F0F9FF, #E0F2FE) !important;
    border-radius: 18px !important;
    border: 1px solid #BAE6FD !important;
    box-shadow: 0 10px 40px rgba(2,132,199,0.25) !important;
    padding: 18px !important;
}

/* text */
div[data-testid="stPopover"] * {
    color: #0F2A44 !important;
}

/* title */
div[data-testid="stPopover"] h3 {
    color: #0284C7 !important;
    font-weight: 700;
}

/* divider */
div[data-testid="stPopover"] hr {
    border-color: #BAE6FD !important;
}

/* ===== FIX POPOVER FULL ===== */

/* container utama */
div[data-testid="stPopover"] > div {
    background: linear-gradient(180deg, #F0F9FF, #E0F2FE) !important;
    border-radius: 18px !important;
    border: 1px solid #BAE6FD !important;
    box-shadow: 0 10px 40px rgba(2,132,199,0.25) !important;
    padding: 18px !important;
}

/* text */
div[data-testid="stPopover"] * {
    color: #0F2A44 !important;
}

/* title */
div[data-testid="stPopover"] h3 {
    color: #0284C7 !important;
}

/* scroll area fix */
section[role="dialog"] {
    background: transparent !important;
}
/* ===== AI INTERPRETATION CARD ===== */
.ai-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 6px 20px rgba(15, 42, 68, 0.06);
}

/* inner content */
.ai-box {
    background: #F8FAFC;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #E2E8F0;
    margin-top: 10px;
    line-height: 1.6;
}       
/* ===== POPOVER BUTTON (DETAILS) ===== */
button[data-testid="stPopoverButton"] {
    background: linear-gradient(135deg, #0284C7, #38BDF8) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 8px 14px !important;
    box-shadow: 0 4px 12px rgba(2,132,199,0.3) !important;
}
/* 🔥 HAPUS CONTAINER POPUP WRAPPER */
div[data-testid="stPopover"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
}

/* container luar tombol */
div[data-testid="stPopover"] > div {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
}
/* jarak tombol ke card */
div[data-testid="stPopoverButton"] {
    margin-top: -10px !important;
}
button[data-testid="stPopoverButton"] {
    width: 100% !important;
    border-radius: 10px !important;
    margin-top: -12px !important;
    transform: translateY(-8px);
}
/* hover */
button[data-testid="stPopoverButton"]:hover {
    opacity: 0.9 !important;
}
/* ===== STICKY FOOTER ===== */
html, body {
    height: 100%;
}

.stApp {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

[data-testid="stAppViewContainer"] {
    flex: 1;
    display: flex;
    flex-direction: column;
}

[data-testid="stAppViewContainer"] > .main {
    flex: 1;
}

/* footer */
.footer {
    margin-top: auto;
    text-align:center;
    padding:14px;
    background: linear-gradient(160deg, #E0F2FE, #BAE6FD, #7DD3FC);
    border-top:1px solid #E0F2FE;
    color:#0F2A44;
    font-size:0.85rem;
}
.white-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 30px rgba(15, 42, 68, 0.08);
    margin-top: 20px;
}

.card-title {
    font-weight: 700;
    font-size: 18px;
    color: #0F2A44;
    margin-bottom: 16px;
}

button[data-baseweb="tab"] {
    font-weight: 600;
    color: #64748B;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #0284C7;
    border-bottom: 2px solid #0284C7;
}
                    
</style>
""", unsafe_allow_html=True)

def set_footer():
    st.markdown("""
    <style>
    .footer-fixed {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(160deg, #E0F2FE, #BAE6FD, #7DD3FC);
        text-align: center;
        padding: 12px;
        font-size: 0.85rem;
        color: #0F2A44;
        border-top: 1px solid #E0F2FE;
        z-index: 999;
    }
    </style>

    <div class="footer-fixed">
        Tropical Cyclone Detection System<br>
        <span style="font-size:0.75rem;color:#64748B;">
            Developed by Andini Nareswari Rifaat
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── State ──
if "page" not in st.session_state:
    st.session_state.page = "home"
if "uploaded" not in st.session_state:
    st.session_state.uploaded = None
if "loading" not in st.session_state:
    st.session_state.loading = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# ══════════════════════════════════════
# PAGE: LOADING PAGE
# ══════════════════════════════════════
if st.session_state.loading:

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        lottie = load_lottie(os.path.join(BASE_DIR, "assets", "loading.json"))
        st_lottie(lottie, height=280)

        st.markdown(
            "<h3 style='text-align:center;color:#0284C7;'>Analyzing Cyclone...</h3>",
            unsafe_allow_html=True
        )

    # ⏳ tampilkan dulu
    if not st.session_state.processing:
        st.session_state.processing = True
        st.rerun()   #  tampilkan UI dulu
        st.stop()

    # ⚙️ proses baru jalan
    image = Image.open(st.session_state.uploaded).convert("RGB")
    img_np = np.array(image)

    mask, boxes, overlay = run_inference(model, img_np)

    st.session_state.result_mask = mask
    st.session_state.boxes = boxes
    st.session_state.result_overlay = overlay

    st.session_state.loading = False
    st.session_state.page = "result"

    st.rerun()

# ══════════════════════════════════════
# PAGE: RESULT
# ══════════════════════════════════════
if st.session_state.page == "result":
    # Header
    st.markdown("""
    <div class="result-header">
        <span style="font-weight:1000;color:#0C4A6E;font-size:3rem;text-align:center;"> DETECTION RESULTS </span>
        <span class="result-badge">Analysis Complete</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    # Image tabs
    img = st.session_state.uploaded

    # ===== TITLE + CARD =====
    st.markdown("""
    <div class="white-card" style="margin-top:1rem;">
        <div class="card-title">Image Visualization</div>
    </div>
    """, unsafe_allow_html=True)

    # ===== ZOOM CONTROL=====
    zoom = st.slider("Zoom (px)", 200, 800, 500)
    fit = st.toggle("Full Width")

    def get_width():
        return "stretch" if fit else zoom

    # ===== TABS =====
    tab1, tab2, tab3 = st.tabs(["Original Image", "Segmentation Mask", "Bounding Box"])

    with tab1:
        if img:
            st.image(img, caption="Original Satellite Image", width=get_width())
        else:
            st.info("No image uploaded.")

    with tab2:
        if img:
            st.image(st.session_state.result_mask, caption="Segmentation Mask", width=get_width())
            st.caption("Segmentation Mask generated by Model DeepLabV3+")
        else:
            st.info("No image uploaded.")

    with tab3:
        if img:
            st.image(st.session_state.result_overlay, caption="Bounding Box Result", width=get_width())
            st.caption("Bounding Box that segmentation model identified cyclone structural area")
        else:
            st.info("No image uploaded.")
    # Bottom panels
    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.session_state.get("selected_date", None)
        if selected_date:
            date_str = selected_date.strftime("%Y-%m-%d")
            period = "am" if selected_date.hour < 12 else "pm"
            zoom_url = f"https://zoom.earth/maps/satellite-hd/#view=-0.5,117.7,4.69z/date={date_str},{period}"
        else:
            zoom_url = "https://zoom.earth"

        st.markdown(f"""
        <div class="white-card">
            <div style="font-size:28px;font-weight:700;color:#0F2A44;margin-bottom:12px;">
                Satellite Validation
            </div>
            <div class="grok-box">
                Cross-reference the detection results with real-time satellite data from Zoom Earth for validation.
                <br><br>
                <a href="{zoom_url}" target="_blank"
                style="display:inline-flex;align-items:center;gap:8px;
                background:linear-gradient(135deg,#0284C7,#38BDF8);
                color:#fff;padding:10px 24px;border-radius:12px;
                text-decoration:none;font-weight:700;
                box-shadow:0 4px 16px rgba(2,132,199,0.35);">
                Open Zoom Earth
                </a>
                <br><br>
                <small style="color:#64748B;">
                    <strong>Tip:</strong> Compare the cyclone position and structure
                    with live satellite feeds to validate detection accuracy.
                </small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        boxes = st.session_state.get("boxes", [])

        if boxes:
            indo, eng = interpret_boxes(
                st.session_state.boxes,
                st.session_state.result_overlay.shape,
                st.session_state.get("selected_date", None)
            )

            formatted_indo = indo.replace("\n", "<br>")
            formatted_eng = eng.replace("\n", "<br>")

        else:
            formatted_indo = "Tidak terdeteksi aktivitas signifikan. Cuaca cenderung cerah."
            formatted_eng = "No significant activity detected. Weather is generally clear."

        # ===== CARD WRAPPER OPEN =====
        st.markdown("""
        <div class="white-card">
            <div style="font-size:28px;font-weight:700;color:#0F2A44;margin-bottom:10px;">
                AI Interpretation
            </div>
        """, unsafe_allow_html=True)

        # ===== TABS (WAJIB DI LUAR HTML) =====
        tab1, tab2 = st.tabs(["🇮🇩 Indonesia", "🇬🇧 English"])

        with tab1:
            st.markdown("#### Indonesian Analysis")
            st.markdown(f"""
            <div class="ai-box">
            {formatted_indo}
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("#### English Analysis")
            st.markdown(f"""
            <div class="ai-box">
            {formatted_eng}
            </div>
            """, unsafe_allow_html=True)

        # ===== CARD WRAPPER CLOSE =====
        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    set_footer()

# ══════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════
else:
    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge"> AI-Powered Meteorological Analysis</div>
        <div class="hero-title">AI-Based Tropical Cyclone<br><span>Detection System</span></div>
        <div class="hero-sub">
            Leveraging U-Net deep learning segmentation and LLM-powered interpretation to detect
            and analyze tropical cyclones from satellite imagery with precision and clarity.
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Info cards
    st.markdown("""
    <style>

    /* biar card keliatan menyatu */
    .step-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #E0F2FE;
        margin-bottom: 8px;
    }

    /* hover effect */
    .step-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(2,132,199,0.2);
    }

    /* number bulat */
    .step-num {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0284C7, #38BDF8);
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 8px;
    }

    /* title */
    .step-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #0F2A44;
    }

    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Understanding the System</div>', unsafe_allow_html=True)

    cards = [
    ("01", "What is a Tropical Cyclone?", "cyclone",
     "A large rotating storm system formed over warm oceans...",
     os.path.join(BASE_DIR, "assets", "1_cyclone.json")),

    ("02", "Pseudo-Mask Generator", "dvorak",
     "Uses Dvorak technique to generate pseudo labels...",
     os.path.join(BASE_DIR, "assets", "2_generator.json")),

    ("03", "Deep Learning Model", "deeplab",
     "DeepLabV3+ for multi-class segmentation...",
     os.path.join(BASE_DIR, "assets", "3_AI.json")),

    ("04", "Structural Localization", "localization",
     "Detect cyclone core and structure...",
     os.path.join(BASE_DIR, "assets", "4_temporal.json")),

    ("05", "LLM Interpretation", "llm",
     "Generate human-readable insights...",
     os.path.join(BASE_DIR, "assets", "5_Chatbot.json")),
    ]

    cols = st.columns(5)

    for col, (num, title, key, desc, lottie_path) in zip(cols, cards):
        with col:

            # ===== CARD VISUAL =====
            with open(lottie_path, "r") as f:
                lottie_json = f.read()

            components.html(f"""
            <div style="
                background:white;
                border-radius:18px;
                padding:18px;
                border:1px solid #E0F2FE;
                text-align:center;
                transition: all 0.25s ease;
            "
            onmouseover="
                this.style.transform='translateY(-6px)';
                this.style.boxShadow='0 0 30px rgba(56,189,248,0.7)';
            "
            onmouseout="
                this.style.transform='none';
                this.style.boxShadow='none';
            "
            >

                <div style="
                    width:32px;height:32px;
                    border-radius:50%;
                    background:linear-gradient(135deg,#0284C7,#38BDF8);
                    color:white;
                    font-weight:bold;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:12px;
                    margin:0 auto 10px auto;
                ">
                    {num}
                </div>

                <div style="
                    font-weight:700;
                    font-size:18px;
                    color:#0F2A44;
                    margin-bottom:10px;
                ">
                    {title}
                </div>

                <div id="lottie-{key}" style="height:110px;"></div>

                <div style="
                    font-size:14px;
                    color:#64748B;
                    margin-top:10px;
                ">
                    {desc}
                </div>

            </div>

            <script src="https://unpkg.com/lottie-web@5.10.2/build/player/lottie.min.js"></script>
            <script>
            lottie.loadAnimation({{
                container: document.getElementById("lottie-{key}"),
                renderer: 'svg',
                loop: true,
                autoplay: true,
                animationData: {lottie_json}
            }});
            </script>
            """, height=300)

            # ===== POPOVER =====
            with st.popover("Details", width="stretch"):

                # ===== CYCLONE =====
                if key == "cyclone":

                    st.markdown("### What is a Tropical Cyclone?")

                    st.image(
                        os.path.join(BASE_DIR, "assets", "siklon_tropis.jpg"),
                        width="stretch"
                    )

                    st.markdown("""
                    Tropical cyclones are large-scale atmospheric systems that develop over warm ocean waters,
                    characterized by a low-pressure center, strong rotating winds, and organized convective cloud structures.
                    """)

                    st.markdown("#### Formation Conditions")
                    st.markdown("""
                    - High sea surface temperatures  
                    - High humidity in the lower and mid-troposphere  
                    - Sufficient Coriolis force  
                    - Low vertical wind shear  
                    - Atmospheric instability  
                    """)

                    st.markdown("---")

                    st.image(
                        os.path.join(BASE_DIR, "assets", "dampak_ST.jpeg"),
                        width="stretch"
                    )

                    st.markdown("#### Impacts : ")
                    st.markdown("""
                    Tropical cyclones can trigger floods, landslides, storm surges, extreme winds, and rough seas,
                    causing significant economic losses and human casualties.
                    """)

                # ===== DVORAK =====
                elif key == "dvorak":

                    st.markdown("### Pseudo-Mask Generator with Dvorak Technique")

                    st.image(
                        os.path.join(BASE_DIR, "assets", "dvorak.png"),
                        width="stretch"
                    )

                    st.markdown("""
                    The Dvorak technique estimates cyclone intensity using satellite imagery
                    based on cloud patterns such as curved bands, CDO, and eye structures.
                    """)

                    st.markdown("#### Processing")
                    st.markdown("""
                    - Image cropping  
                    - BGR → HSV transformation  
                    - Structural thresholding  
                    """)

                    st.markdown("#### Parameters")
                    st.markdown("""
                    - Convective core area  
                    - Active sector ratio  
                    - Band distribution  
                    - Symmetry ratio  
                    """)

                    st.markdown("""
                    Output: multi-class pseudomask (Red Core, Impacted Area, DCC, Background)
                    """)

                # ===== DEEPLAB =====
                elif key == "deeplab":
                    st.markdown("### DeepLabV3+ Model")

                    st.markdown("""
                    DeepLabV3+ is a semantic segmentation model that utilizes atrous convolution 
                    and an encoder–decoder structure to capture multi-scale features.
                    """)

                    # ===== TABLE DATA =====
                    data = {
                        "Class": ["Background", "Red Core", "Impacted Area", "DCC", "Mean"],
                        "Detection Accuracy": ["-", "0.836", "0.832", "0.603", "-"],
                        "Precision (Cyclone)": [0.9956, 0.6515, 0.5570, 0.4054, 0.6524],
                        "Recall (Cyclone)": [0.9940, 0.8149, 0.6422, 0.0485, 0.6249],
                        "F1-score (Cyclone)": [0.9948, 0.7241, 0.5966, 0.0866, 0.6005],
                        "IoU (Cyclone)": [0.9896, 0.5675, 0.4251, 0.0452, 0.5069],
                    }

                    df = pd.DataFrame(data)

                    # ===== DISPLAY TABLE =====
                    df = df.astype(str)
                    st.dataframe(df, width='stretch')

                    st.markdown("""
                    ### Analysis

                    Strong performance at detecting:
                    - Red Core
                    - Impacted Area  

                    Weak performance at detecting:
                    - DCC (small, scattered objects)

                    The model performs well in detecting main cyclone structures but struggles 
                    with small-scale patterns and clear-sky false positives.
                    """)

                # ===== LOCALIZATION =====
                elif key == "localization":

                    st.markdown("### Structural Localization")

                    st.image(
                        os.path.join(BASE_DIR, "assets", "localization.png"),
                        width="stretch"
                    )

                    st.markdown("""
                    Segmentation results are converted into bounding boxes
                    representing cyclone structures.
                    """)

                    st.markdown("""
                    - Red Core → cyclone center  
                    - Impacted Area → surrounding region  
                    - DCC → unstructured clouds  
                    """)

                    st.markdown("""
                    Filtering and convex hull improve spatial accuracy.
                    """)

                # ===== LLM =====
                elif key == "llm":

                    st.markdown("### LLM Interpretation")

                    st.markdown("""
                    The LLM generates structured summaries from bounding box outputs. LLM using model from Open Router with failover between google/gemma-3-4b-it and qwen/qwen3.6-plus free tiers. For last option model wouldd be random pick from openrouter/free.
                    """)

                    st.markdown("""
                    Input:
                    - Number of detections  
                    - Spatial distribution  
                    - Class types  
                    """)

                    st.markdown("""
                    Output:
                    - Weather condition  
                    - Cyclone potential  
                    - Risk analysis  
                    """)

                    st.markdown("""
                    Notes : LLM acts only as narrator and All logic from CV pipeline  
                    """)

            

                
    # How to use
    st.markdown('<div class="section-title">How to Use</div>', unsafe_allow_html=True)

    steps = [
        ("Upload Satellite Image", "Select a PNG satellite image of the target region for analysis."),
        ("Select Timestamp", "Choose the date corresponding to the satellite capture for validation."),
        ("Click \"Detect Cyclone\"", "Run the AI model to segment and classify cyclone structures in the image."),
    ]

    # Build HTML string (TANPA indent aneh)
    steps_html = ""
    for i, (title, desc) in enumerate(steps, 1):
        steps_html += f"""
    <div class="step-row">
        <div class="step-num">{i}</div>
        <div class="step-text">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
    </div>
    """

    # Render dalam SATU block (ini kunci utama)
    html_content = f"""
    <div class="white-card" style="max-width:720px;margin:0 auto;">
    {steps_html}
    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)
    # ── Input Section ──
    st.markdown('<div class="section-title">Start Detection</div>', unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])[1]

    with col_center:
        with st.container():

            st.markdown('<div class="input-label"> Satellite Image (PNG)</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload PNG image",
                type=["png"],
                label_visibility="collapsed",
                key="main_uploader"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="input-label"> Capture Date</div>', unsafe_allow_html=True)
            selected_date = st.datetime_input(
                "Select date",
                value=datetime.now(),
                label_visibility="collapsed",
                key="date_input"
            )

            st.session_state.selected_date = selected_date

            st.markdown("<br>", unsafe_allow_html=True)

            # 🔥 FIX BUTTON → TRIGGER LOADING
            if st.button(" Detect Cyclone", disabled=uploaded is None, key="detect_btn"):

                # simpan dulu data
                st.session_state.uploaded = uploaded
                st.session_state.selected_date = selected_date

                # aktifkan loading mode
                st.session_state.loading = True
                st.session_state.processing = False

                # rerun → pindah ke loading screen
                st.rerun()

    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    set_footer()


# st.markdown("""
# <div class="footer">
#     Tropical Cyclone Detection System  
#     <br>
#     <span style="font-size:0.75rem;color:#64748B;">
#         Developed by Andini Nareswari Rifaat
#     </span>
# </div>
# """, unsafe_allow_html=True)
