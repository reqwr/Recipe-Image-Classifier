import streamlit as st
from pathlib import Path
from PIL import Image
import os
import tempfile
import numpy as np
import pandas as pd
import hashlib

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from huggingface_hub import hf_hub_download

from src.utils import load_keras, load_joblib, load_faiss, load_npy

# ============================================================
# CONFIG
# ============================================================
DATASET_REPO = "cyuangli/WebEats-v3"
EMBEDDING_MODEL_REPO = "cyuangli/embedding-model"
FAISS_REPO = "cyuangli/recipe-faiss"
IMAGE_DATA_REPO = "cyuangli/image-data"
PCA_REPO = "cyuangli/pca-data"

IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="WebEats - Recipe Image Search",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION STATE
# ============================================================
if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = None

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    .image-grid {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 14px;
        margin-top: 20px;
    }
    .stImage {
        cursor: pointer;
        transition: transform 0.15s ease-in-out;
    }
    .stImage:hover {
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def extract_lemmatized_name(image_path: str) -> str:
    filename = Path(image_path).stem
    parts = filename.split("_")
    name = []
    for p in parts:
        if p.isdigit():
            break
        name.append(p)
    return " ".join(name)


@st.cache_data
def load_recipe_csv():
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="recipe_meta_topics.csv",
        repo_type="dataset",
    )
    return pd.read_csv(path)


def get_recipe_data(lemmatized_name: str):
    df = load_recipe_csv()
    row = df[df["lemmatized_name"] == lemmatized_name]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "original_name": row["original_name"],
        "recipe": row["recipe"],
    }


@st.cache_data
def get_image_local(image_path: str):
    if os.path.exists(image_path):
        return image_path
    alt = os.path.join("notebooks", image_path)
    if os.path.exists(alt):
        return alt
    return None

# ============================================================
# MODEL LOADING (INTENTIONALLY NOT CACHED)
# ============================================================
def load_models():
    embedding_model_path = hf_hub_download(
        repo_id=EMBEDDING_MODEL_REPO,
        filename="embedding_model.keras",
        repo_type="model",
        force_download=True,
    )

    pca_path = hf_hub_download(
        repo_id=PCA_REPO,
        filename="pca.joblib",
        repo_type="model",
    )

    faiss_path = hf_hub_download(
        repo_id=FAISS_REPO,
        filename="recipes.faiss",
        repo_type="model",
    )

    image_paths_path = hf_hub_download(
        repo_id=IMAGE_DATA_REPO,
        filename="image_paths.npy",
        repo_type="model",
    )

    embedding_model = load_keras(embedding_model_path)
    pca = load_joblib(pca_path)
    index = load_faiss(faiss_path)
    image_paths = load_npy(image_paths_path)

    return embedding_model, pca, index, image_paths


embedding_model, pca, index, image_paths = load_models()

# ============================================================
# EMBEDDING + SEARCH
# ============================================================
def embed_image(image_path: str) -> np.ndarray:
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    emb = embedding_model(arr, training=False).numpy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def search_similar_recipes(image_path: str, k: int = 25):
    emb = embed_image(image_path)
    emb_pca = pca.transform(emb)
    query = emb_pca.astype("float32")

    distances, indices = index.search(query, k)
    return image_paths[indices[0]], distances[0]

# ============================================================
# UI
# ============================================================
st.title("🍲 WebEats")
st.caption("Find similar recipes by uploading a food image")

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
)

# ============================================================
# MAIN FLOW
# ============================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Your Image")
    st.image(image, width=350)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        query_path = tmp.name

    results, distances = search_similar_recipes(query_path, k=25)
    os.unlink(query_path)

    st.subheader("🔍 Similar Recipes")

    grid_col, detail_col = st.columns([2, 1], gap="large")

    # ========================================================
    # IMAGE GRID
    # ========================================================
    with grid_col:
        st.markdown('<div class="image-grid">', unsafe_allow_html=True)
        cols = st.columns(5)
        for i, path in enumerate(results):
            img_path = get_image_local(path)
            if not img_path:
                continue

            with cols[i % 5]:
                st.image(img_path, use_container_width=True)
                if st.button("View Recipe", key=f"view_{i}"):
                    st.session_state.selected_recipe = path
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # RECIPE SIDE PANEL (NO DIMMING)
    # ========================================================
    with detail_col:
        if st.session_state.selected_recipe:
            recipe_path = st.session_state.selected_recipe
            data = get_recipe_data(
                extract_lemmatized_name(recipe_path)
            )

            if data:
                st.markdown("### 🍽 Recipe Details")
                st.image(
                    get_image_local(recipe_path),
                    use_container_width=True,
                )

                st.markdown(f"**{data['original_name']}**")
                st.markdown("---")

                for step in data["recipe"].split("|"):
                    step = step.strip()
                    if step:
                        st.markdown(step)
