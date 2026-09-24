import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.set_page_config(page_title="Lector de Texto", page_icon="🔍", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 20%, #312e81 0%, #1e1b4b 60%, #0f0f23 100%);
    color: #e2e8f0;
}

/* Header */
.hero {
    text-align: center;
    padding: 1rem 0 2rem 0;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Result card */
.result-card {
    background: linear-gradient(145deg, #ffffff, #f1f5f9);
    border-radius: 20px;
    padding: 1.75rem 2rem;
    color: #0f172a;
    font-size: 1.05rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1);
    margin-top: 1rem;
}

/* Section labels */
.section-label {
    color: #c7d2fe;
    font-weight: 600;
    font-size: 1.1rem;
    margin: 1.5rem 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Buttons */
.stDownloadButton > button, .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.6) !important;
}

/* Camera input */
[data-testid="stCameraInput"] {
    border-radius: 20px;
    overflow: hidden;
}

/* Hide streamlit menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔍 Lector de Texto</h1>
    <p>Toma una foto y el texto aparecerá como por arte de magia</p>
</div>
""", unsafe_allow_html=True)


def preprocess(img):
    """Convierte a escala de grises, auto-invierte si es necesario y aplica umbral adaptativo."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Auto-invertir si el fondo es oscuro
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    
    # Suavizar un poco para reducir ruido
    gray = cv2.medianBlur(gray, 3)
    
    # Umbral adaptativo: excelente para fotos con luz desigual
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )
    return thresh


img_file_buffer = st.camera_input("📷 Captura una imagen")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-label">🖼️ Imagen procesada</div>', unsafe_allow_html=True)
        processed = preprocess(cv2_img)
        st.image(processed, use_container_width=True, clamp=True)
    
    with col2:
        st.markdown('<div class="section-label">📝 Texto reconocido</div>', unsafe_allow_html=True)
        with st.spinner("Analizando..."):
            text = pytesseract.image_to_string(processed)
        
        if text.strip():
            st.markdown(f'<div class="result-card">{text.strip()}</div>', unsafe_allow_html=True)
        else:
            st.warning("No se detectó texto. Prueba con más luz o acerca la cámara.")
    
    if text.strip():
        st.download_button(
            "⬇️ Descargar texto",
            data=text,
            file_name="texto_reconocido.txt",
            mime="text/plain",
            use_container_width=True
        )
