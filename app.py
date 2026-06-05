import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(
    page_title="PCA Analysis",
    layout="wide"
)

st.title("Principal Component Analysis - Spotify Dataset")

st.header("1. Data Ingestion")

@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(
            RAW_DIR,
            "dataset.csv"
        )
    )

df = load_data()

st.success("Dataset Loaded Successfully")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.header("2. Data Cleaning")

strategy = st.selectbox(
    "Missing Value Strategy",
    [
        "Mean",
        "Median",
        "Most Frequent",
        "Drop Rows"
    ]
)

df_clean = df.copy()

if strategy == "Drop Rows":

    df_clean = df_clean.dropna()

else:

    fill_map = {
        "Mean": "mean",
        "Median": "median",
        "Most Frequent": "most_frequent"
    }

    imputer = SimpleImputer(
        strategy=fill_map[strategy]
    )

    numeric_cols = df_clean.select_dtypes(
        include=np.number
    ).columns

    df_clean[numeric_cols] = imputer.fit_transform(
        df_clean[numeric_cols]
    )

st.dataframe(
    df_clean.head(),
    use_container_width=True
)

if st.button("Save Cleaned Dataset"):

    clean_path = os.path.join(
        CLEAN_DIR,
        "spotify_cleaned.csv"
    )

    df_clean.to_csv(
        clean_path,
        index=False
    )

    st.success(
        "Dataset Saved Successfully"
    )

st.header("3. Load Cleaned Dataset")

files = [
    f
    for f in os.listdir(CLEAN_DIR)
    if f.endswith(".csv")
]

if not files:

    st.warning(
        "No cleaned dataset found"
    )

    st.stop()

file = st.selectbox(
    "Select Dataset",
    files
)

data = pd.read_csv(
    os.path.join(
        CLEAN_DIR,
        file
    )
)

st.dataframe(
    data.head(),
    use_container_width=True
)

features = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

available_features = [
    col for col in features
    if col in data.columns
]

data = data[available_features]

st.sidebar.header(
    "PCA Settings"
)

max_components = min(
    len(available_features),
    10
)

n_components = st.sidebar.slider(
    "Number Of Components",
    2,
    max_components,
    2
)

X = data.copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

st.header("4. PCA Training")

pca = PCA(
    n_components=n_components
)

X_pca = pca.fit_transform(
    X_scaled
)

st.success(
    "PCA Completed Successfully"
)

st.header("5. Model Information")

st.subheader("Model")

st.write(
    type(pca).__name__
)

params_df = pd.DataFrame(
    list(pca.get_params().items()),
    columns=[
        "Parameter",
        "Value"
    ]
)

st.dataframe(
    params_df,
    use_container_width=True
)

st.header(
    "6. Explained Variance Ratio"
)

variance_df = pd.DataFrame({
    "Component": [
        f"PC{i+1}"
        for i in range(n_components)
    ],
    "Explained Variance":
        pca.explained_variance_ratio_
})

st.dataframe(
    variance_df,
    use_container_width=True
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

sns.barplot(
    data=variance_df,
    x="Component",
    y="Explained Variance",
    ax=ax
)

ax.set_title(
    "Explained Variance Ratio"
)

st.pyplot(fig)

st.header(
    "7. Cumulative Explained Variance"
)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

cum_df = pd.DataFrame({
    "Component": [
        f"PC{i+1}"
        for i in range(n_components)
    ],
    "Cumulative Variance":
        cumulative_variance
})

st.dataframe(
    cum_df,
    use_container_width=True
)

fig2, ax2 = plt.subplots(
    figsize=(8, 5)
)

ax2.plot(
    range(
        1,
        n_components + 1
    ),
    cumulative_variance,
    marker="o"
)

ax2.set_xlabel(
    "Number Of Components"
)

ax2.set_ylabel(
    "Cumulative Variance"
)

ax2.set_title(
    "Cumulative Explained Variance"
)

st.pyplot(fig2)

st.header(
    "8. PCA Components"
)

components_df = pd.DataFrame(
    pca.components_,
    columns=available_features,
    index=[
        f"PC{i+1}"
        for i in range(n_components)
    ]
)

st.dataframe(
    components_df,
    use_container_width=True
)

st.header(
    "9. PCA Transformed Dataset"
)

pca_columns = [
    f"PC{i+1}"
    for i in range(n_components)
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

st.dataframe(
    pca_df.head(20),
    use_container_width=True
)

st.header(
    "10. PCA Scatter Plot"
)

if n_components >= 2:

    fig3, ax3 = plt.subplots(
        figsize=(8, 5)
    )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        ax=ax3
    )

    ax3.set_title(
        "PCA Projection"
    )

    st.pyplot(fig3)

st.header(
    "11. Save Model"
)

model_path = os.path.join(
    MODEL_DIR,
    "pca_model.pkl"
)

with open(
    model_path,
    "wb"
) as f:

    pickle.dump(
        pca,
        f
    )

st.success(
    f"Model Saved At : {model_path}"
)

st.header(
    "12. Save PCA Dataset"
)

pca_path = os.path.join(
    CLEAN_DIR,
    "spotify_pca.csv"
)

pca_df.to_csv(
    pca_path,
    index=False
)

st.success(
    f"PCA Dataset Saved At : {pca_path}"
)

st.dataframe(
    pca_df.head(),
    use_container_width=True
)