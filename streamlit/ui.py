import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

st.title("Comptage de colonies dans des boites de Petri")
st.sidebar.image("logo/Logo_CITC_Gris.png", use_column_width=True)

# interact with FastAPI endpoint
st.sidebar.header("Sélection du type d'analyse")
genre = st.sidebar.radio(
    "Sélectionnez votre type d'analyse", ("Segmentation", "Détection")
)

if genre == "Segmentation":
    backend = "http://fastapi_ecco:8000/segmentation"
else:
    backend = "http://fastapi_ecco:8000/detection"


def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url,
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000,
    )

    return r


# construct UI layout

st.write(
    """Obtenez la segmentation sémantique/détection de l'image chargée grâce aux modèles ResNet50v2-FPN/YOLOv5 implémentés. Visitez l'URL à `:8000/docs` pour la documentation FastAPI."""
)  # description and instructions

input_image = st.file_uploader("insérez une image")  # image upload widget

if st.sidebar.button("Lancer l'analyse"):

    st.markdown("## Visualisation")

    col1, col2 = st.columns(2)

    if input_image:
        segments = process(input_image, backend)
        original_image = Image.open(input_image)
        result_image = Image.open(io.BytesIO(segments.content))
        col1.header("Image")
        original_image = original_image.resize((1024, 1024))
        col1.image(original_image, use_column_width=True)
        col2.header("Résultat")
        col2.image(result_image, use_column_width=True)

    else:
        # handle case with no image
        st.write("Insérez an image!")
