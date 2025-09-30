import os
import io
import numpy as np
import streamlit as st
from PIL import Image

# === TensorFlow / Keras (TF 2.15) ===
# Import late to make error messages amigáveis se faltar dependência
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    st.error("Falha ao importar TensorFlow/Keras. "
             "Verifique se o deploy inclui 'tensorflow-cpu==2.15.0' no requirements.txt "
             "e se o Python é 3.10 (runtime.txt). Detalhes: " + str(e))
    st.stop()

st.set_page_config(page_title="Folhas de Café - Classificação", layout="wide")

st.title("☕ Classificação de Doenças em Folhas de Café")
st.write("Faça **upload** de uma ou várias imagens para obter a predição do modelo.")

# ===== Config do modelo ====
CLASSES = ['healthy', 'miner', 'cerscospora', 'phoma', 'leaf_rust']
IMG_SIZE = int(os.getenv("IMG_SIZE", "100"))
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_folhas_cafe.keras")

# ===== Tentativa de carregar o modelo localmente, senão baixar do Hugging Face Hub =====
_model = None
def get_model():
    global _model
    if _model is not None:
        return _model

    path = MODEL_PATH
    if not os.path.exists(path):
        # Tenta baixar se variáveis HF_REPO e HF_FILENAME estiverem configuradas
        hf_repo = os.getenv("HF_REPO")  # exemplo: "seu-usuario/seu-modelo"
        hf_filename = os.getenv("HF_FILENAME")  # exemplo: "modelo_folhas_cafe.keras"
        if hf_repo and hf_filename:
            try:
                from huggingface_hub import hf_hub_download
                st.info(f"Baixando modelo de {hf_repo}/{hf_filename}...")
                path = hf_hub_download(repo_id=hf_repo, filename=hf_filename, local_dir=".", local_dir_use_symlinks=False)
            except Exception as e:
                st.error(f"Não foi possível baixar o modelo do Hugging Face Hub: {e}")
                st.stop()
        else:
            st.error(f"Modelo '{MODEL_PATH}' não encontrado no repositório. "
                     "Envie o arquivo para a raiz do projeto ou configure HF_REPO e HF_FILENAME nas Secrets.")
            st.stop()

    try:
        _model = load_model(path, compile=False)
    except Exception as e:
        st.error(f"Falha ao carregar o modelo em '{path}': {e}")
        st.stop()
    return _model

def preprocess_pil(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(pil_img: Image.Image):
    model = get_model()
    x = preprocess_pil(pil_img)
    probs = model.predict(x, verbose=0)[0].tolist()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

# ===== Interface =====
tab1, tab2 = st.tabs(["Imagem única", "Lote (múltiplos arquivos)"])

with tab1:
    up = st.file_uploader("Envie uma imagem (jpg/png)", type=["jpg","jpeg","png"], accept_multiple_files=False)
    if up is not None:
        img = Image.open(up)
        st.image(img, caption="Imagem enviada", use_column_width=True)
        idx, probs = predict_image(img)
        st.subheader(f"Predição: **{CLASSES[idx]}**")
        st.write({c: float(p) for c, p in zip(CLASSES, probs)})

with tab2:
    ups = st.file_uploader("Envie várias imagens (jpg/png)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if ups:
        import pandas as pd
        rows = []
        cols = ["arquivo", "predicao"] + CLASSES
        progress = st.progress(0)
        for i, upf in enumerate(ups, start=1):
            try:
                img = Image.open(upf)
                idx, probs = predict_image(img)
                row = {"arquivo": upf.name, "predicao": CLASSES[idx]}
                row.update({c: float(p) for c, p in zip(CLASSES, probs)})
                rows.append(row)
            except Exception as e:
                rows.append({"arquivo": upf.name, "predicao": f"erro: {e}"})
            progress.progress(i/len(ups))
        df = pd.DataFrame(rows, columns=cols)
        st.dataframe(df, use_container_width=True)
        st.download_button("Baixar resultados (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="resultados.csv", mime="text/csv")

st.caption("Dica: se o modelo for maior que 100MB, suba para o Hugging Face Hub e configure as variáveis HF_REPO e HF_FILENAME em *Secrets*.")
