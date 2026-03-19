# =========================================
# BRCA Multi-Omics AI — Breast Cancer Subtyping & Biomarker Intelligence Platform
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Load artifacts (DEPLOYMENT SAFE)
# -------------------------------
@st.cache_resource
def load_artifacts():
    base_path = "models"  # <-- IMPORTANT

    model = pickle.load(open(os.path.join(base_path, "multiomics_xgb_model.pkl"), "rb"))
    features = pickle.load(open(os.path.join(base_path, "features.pkl"), "rb"))
    le = pickle.load(open(os.path.join(base_path, "label_encoder.pkl"), "rb"))

    try:
        scaler = pickle.load(open(os.path.join(base_path, "scaler.pkl"), "rb"))
    except:
        scaler = None

    return model, features, le, scaler


model, features, le, scaler = load_artifacts()

# -------------------------------
# SHAP EXPLAINER (FIXED CACHE)
# -------------------------------
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# -------------------------------
# PATHWAY CACHE
# -------------------------------
@st.cache_data
def run_enrichr(genes):
    enr = gp.enrichr(
        gene_list=genes,
        gene_sets="KEGG_2021_Human",
        organism="human",
        outdir=None,
    )
    return enr.results.head(10)


# -------------------------------
# MODEL INFO
# -------------------------------
MODEL_INFO = {
    "Model": "XGBoost Multi-Omics Classifier",
    "Dataset": "TCGA BRCA",
    "Accuracy": 0.922,
    "Macro F1": 0.861,
    "Weighted F1": 0.912,
}

# -------------------------------
# SAMPLE DATA
# -------------------------------
def generate_sample_data(n=5):
    return pd.DataFrame(np.random.rand(n, len(features)), columns=features)


# Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Sample Dataset")

sample_df = generate_sample_data(5)

st.sidebar.download_button(
    "Download Sample CSV",
    sample_df.to_csv(index=False),
    "sample_multiomics_input.csv",
)

st.sidebar.caption("Use this format to upload your data")

# -------------------------------
# UTILITIES
# -------------------------------
def extract_gene(f):
    f = str(f)
    if f.startswith("RS_"):
        f = f[3:]

    gene = f.split("_")[-1].upper()

    alias_map = {"HER2": "ERBB2", "ER": "ESR1"}
    return alias_map.get(gene, gene)


def clean_gene_list(genes):
    genes = [g.upper() for g in genes]
    return list(set([g for g in genes if g.isalnum() and len(g) > 2]))


def validate_input(df):
    df = df.reindex(columns=features)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median())

    if scaler:
        df = scaler.transform(df)
        df = pd.DataFrame(df, columns=features)

    return df


# -------------------------------
# UI
# -------------------------------
st.set_page_config(layout="wide")

st.title("BRCA Multi-Omics AI — Breast Cancer Subtyping & Biomarker Intelligence")

st.caption(
    "Interpretable multi-omics AI platform for subtype prediction, biomarker discovery, "
    "and pathway-level insights in breast cancer."
)

# Sidebar Model Info
st.sidebar.header("Model Performance")
st.sidebar.metric("Accuracy", "92.2%")
st.sidebar.metric("Macro F1", "0.86")
st.sidebar.metric("Weighted F1", "0.91")

module = st.sidebar.radio(
    "Module",
    [
        "Upload",
        "Prediction",
        "Biomarkers",
        "Pathways",
        "Hub Genes",
    ],
)

# -------------------------------
# Upload
# -------------------------------
if module == "Upload":

    st.info("Upload a CSV file with the same feature columns as the model.")

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        missing_cols = set(features) - set(df.columns)

        if missing_cols:
            st.error(f"Missing {len(missing_cols)} required columns")
        else:
            st.session_state["raw"] = validate_input(df)
            st.success("Data validated")

# -------------------------------
# Prediction
# -------------------------------
if module == "Prediction":

    if "raw" not in st.session_state:
        st.warning("Upload data first")

    else:
        df = st.session_state["raw"]

        if st.button("Run Prediction"):
            preds = model.predict(df)
            probs = model.predict_proba(df)
            labels = le.inverse_transform(preds)

            st.session_state["processed"] = df

            st.subheader("Predicted Subtypes")
            st.dataframe(pd.DataFrame({"Subtype": labels}))

            st.subheader("Prediction Probabilities")
            st.dataframe(pd.DataFrame(probs, columns=le.classes_))

# -------------------------------
# Biomarkers
# -------------------------------
if module == "Biomarkers":

    if "processed" not in st.session_state:
        st.warning("Run prediction first")

    else:
        df = st.session_state["processed"]

        st.subheader("Global Biomarkers")

        imp = model.feature_importances_
        df_imp = pd.DataFrame({"Feature": features, "Importance": imp})
        df_imp["Gene"] = df_imp["Feature"].apply(extract_gene)

        st.dataframe(df_imp.sort_values("Importance", ascending=False).head(30))

        st.subheader("Subtype Biomarkers")

        sample = df.sample(min(50, len(df)))
        explainer = get_explainer(model)
        shap_vals = explainer.shap_values(sample)

        selected = st.selectbox("Subtype", le.classes_)
        idx = list(le.classes_).index(selected)

        shap_cls = np.abs(shap_vals[:, :, idx]).mean(axis=0)

        df_sub = pd.DataFrame({"Feature": features, "Importance": shap_cls})
        df_sub["Gene"] = df_sub["Feature"].apply(extract_gene)

        st.dataframe(df_sub.sort_values("Importance", ascending=False).head(30))

# -------------------------------
# Pathways
# -------------------------------
if module == "Pathways":

    if "processed" not in st.session_state:
        st.warning("Run prediction first")

    else:
        imp = model.feature_importances_

        df_imp = pd.DataFrame({"Feature": features, "Importance": imp})
        df_imp["Gene"] = df_imp["Feature"].apply(extract_gene)

        st.subheader("Global Pathways")

        genes = clean_gene_list(
            df_imp.sort_values("Importance", ascending=False)["Gene"].head(40)
        )

        st.dataframe(run_enrichr(genes))

# -------------------------------
# Hub Genes
# -------------------------------
if module == "Hub Genes":

    st.subheader("Hub Genes (Top Biomarkers)")

    imp = model.feature_importances_

    df_imp = pd.DataFrame({"Feature": features, "Importance": imp})
    df_imp["Gene"] = df_imp["Feature"].apply(extract_gene)

    gene_imp = (
        df_imp.groupby("Gene")["Importance"]
        .sum()
        .reset_index()
        .sort_values("Importance", ascending=False)
    )

    st.dataframe(gene_imp.head(20))

    plt.figure(figsize=(8, 6))
    plt.barh(gene_imp.head(20)["Gene"], gene_imp.head(20)["Importance"])
    plt.gca().invert_yaxis()

    st.pyplot(plt)