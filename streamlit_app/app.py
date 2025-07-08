import io
import os

import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# URL de l'API
API_URL = os.getenv("API_URL", "http://localhost:9500")
# API_URL = f"http://api:{os.getenv('FASTAPI_PORT', '9500')}"

st.set_page_config(
    page_title="Reconnaissance de chiffres manuscrits",
    layout="centered",
)

# --- Titre ---
st.markdown(
    "<h1 style='text-align: center;'>🖍️ Reconnaissance de chiffres manuscrits</h1>",
    unsafe_allow_html=True,
)

# --- Vérification de l'API ---
st.subheader("🔌 État de l'API")
try:
    res = requests.get(f"{API_URL}/health", timeout=2)
    if res.status_code == 200:
        if res.json().get("model_loaded"):
            st.success("✅ API en ligne et modèle chargé")
        else:
            st.warning("⚠️ API en ligne mais modèle manquant")
    else:
        st.error("❌ API inaccessible")
except Exception as e:
    st.error(f"🚫 Erreur de connexion : {e}")

# --- Interface principale ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("✏️ Dessine un chiffre")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("🔍 Prédiction de ton dessin")

    if canvas.image_data is not None:
        # Préparation de l'image
        img = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
        img = img.convert("L").resize((28, 28))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        files = {"file": ("canvas.png", image_bytes, "image/png")}

        if st.button("🔮 Prédire"):
            try:
                response = requests.post(f"{API_URL}/predict", files=files)
                response.raise_for_status()
                pred = response.json()["prediction"]
                st.session_state.prediction = pred
                st.session_state.image_bytes = image_bytes
                st.success(f"🧠 Prédiction : **{pred}**")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

# --- Correction ---
if "prediction" in st.session_state:
    st.markdown("---")
    st.subheader("✏️ Corriger la prédiction")
    st.write(f"Prédiction actuelle : **{st.session_state.prediction}**")
    label = st.selectbox("Quel est le bon chiffre ?", list(range(10)))
    if st.button("📬 Envoyer la correction"):
        files = {"file": ("canvas.png", st.session_state.image_bytes, "image/png")}
        data = {"label": str(label)}
        try:
            response = requests.post(f"{API_URL}/correct", files=files, data=data)
            response.raise_for_status()
            st.success("✅ Correction envoyée ! Merci 🙌")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'envoi : {e}")
