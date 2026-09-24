import html
import cv2
import numpy as np
import pytesseract
import streamlit as st

# ---------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Lector de Texto",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 10%, #312e81 0%, #1e1b4b 55%, #0b0b1a 100%);
    color: #e2e8f0;
}

/* Hero */
.hero { text-align: center; padding: 1rem 0 2rem 0; }
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: 1.05rem; margin-top: .5rem; }

/* Etiquetas de sección */
.section-label {
    color: #c7d2fe;
    font-weight: 600;
    font-size: 1.05rem;
    margin: .25rem 0 .75rem 0;
}

/* Tarjeta de resultado */
.result-card {
    background: linear-gradient(145deg, #ffffff, #f1f5f9);
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    color: #0f172a;
    font-size: 1.05rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: 0 20px 60px rgba(0,0,0,.45), 0 0 0 1px rgba(255,255,255,.08);
}

/* Botones */
.stDownloadButton > button, .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .65rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 24px rgba(99,102,241,.45) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 34px rgba(99,102,241,.65) !important;
}

/* Widget de cámara con esquinas redondeadas */
[data-testid="stCameraInput"] { border-radius: 18px; overflow: hidden; }

/* Ocultar menú y footer de Streamlit */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🔍 Lector de Texto</h1>
    <p>Toma una foto y el texto aparecerá como por arte de magia ✨</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Preprocesado automático
# ---------------------------------------------------------------
def preprocess(img: np.ndarray) -> np.ndarray:
    """Escala de grises + auto-inversión + umbral adaptativo."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Si la imagen es oscura (texto claro sobre fondo negro) -> invertir
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Suavizado ligero para eliminar ruido de la cámara
    gray = cv2.medianBlur(gray, 3)

    # Umbral adaptativo: maneja bien iluminación desigual
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=15,
    )

# ---------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------
img_file_buffer = st.camera_input("📷 Captura una imagen")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    processed = preprocess(cv2_img)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-label">🖼️ Imagen procesada</div>',
                    unsafe_allow_html=True)
        st.image(processed, use_container_width=True, clamp=True)

    with col2:
        st.markdown('<div class="section-label">📝 Texto reconocido</div>',
                    unsafe_allow_html=True)
        with st.spinner("Analizando…"):
            text = pytesseract.image_to_string(processed).strip()

        if text:
            safe = html.escape(text)  # evita que el texto rompa el HTML
            st.markdown(f'<div class="result-card">{safe}</div>',
                        unsafe_allow_html=True)
        else:
            st.warning("No se detectó texto. Prueba con más luz o acércate al texto.")

    if text:
        st.write("")  # pequeño respiro visual
        st.download_button(
            "⬇️  Descargar texto",
            data=text,
            file_name="texto_reconocido.txt",
            mime="text/plain",
            use_container_width=True,
        )
