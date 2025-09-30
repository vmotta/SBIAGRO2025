
# Streamlit App — Coffee Leaf Classifier (fix)

## Por que este pacote?
Resolve o erro `from tensorflow.keras.models import load_model` no Streamlit Cloud usando:
- `tensorflow-cpu==2.15.0` (compatível e leve)
- `runtime.txt` com `python-3.10`

## Como publicar no Streamlit Community Cloud
1. Crie um repositório público no GitHub e envie:
   - `streamlit_app.py`
   - `requirements.txt`
   - `runtime.txt`
   - (opcional) `modelo_folhas_cafe.keras` — se for <= 100MB
2. No Streamlit Cloud: **New app** → selecione o repo/branch → `streamlit_app.py`.
3. Se o modelo for >100MB, **não** coloque no Git:
   - Envie para o **Hugging Face Hub** (repo de modelos) e configure as *Secrets* no Streamlit:
     - `HF_REPO`: por ex. `seu-usuario/seu-modelo`
     - `HF_FILENAME`: por ex. `modelo_folhas_cafe.keras`
   - O app baixa o arquivo automaticamente com cache.

## Variáveis opcionais
- `MODEL_PATH`: caminho do arquivo local (padrão: `modelo_folhas_cafe.keras`)
- `IMG_SIZE`: tamanho do redimensionamento (padrão: `100`)

