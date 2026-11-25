"""Streamlit dashboard for prediction, EDA, and patient segmentation."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from src.features import FEATURE_NAMES

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "heart_model.joblib"
RANDOM_SEED = 42
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="UCI Heart Disease Explorer", page_icon="❤️", layout="wide")


@st.cache_data(show_spinner="Fetching the UCI Cleveland dataset...")
def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=45)
    if dataset.data is None:
        raise ValueError("UCI did not return feature and target data.")
    features = dataset.data.features.loc[:, FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    )
    target = pd.to_numeric(dataset.data.targets["num"], errors="coerce")
    if target.isna().any():
        raise ValueError("The UCI target contains missing or non-numeric values.")
    return features, target.gt(0).astype(int).rename("has_heart_disease")


@st.cache_resource
def load_model_artifact() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def cluster_patients(
    feature_frame: pd.DataFrame, target: pd.Series, n_clusters: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputed = SimpleImputer(strategy="median").fit_transform(feature_frame)
    scaled = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
    transformed = pca.fit_transform(scaled)
    labels = KMeans(
        n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10
    ).fit_predict(transformed)
    plotted = pd.DataFrame(
        {
            "PC1": transformed[:, 0],
            "PC2": transformed[:, 1],
            "cluster": labels,
            "has_heart_disease": target.to_numpy(),
        }
    )
    profile = feature_frame.assign(
        cluster=labels, has_heart_disease=target.to_numpy()
    )
    summary = (
        profile.groupby("cluster")
        .agg(
            patients=("age", "size"),
            average_age=("age", "mean"),
            average_blood_pressure=("trestbps", "mean"),
            average_cholesterol=("chol", "mean"),
            heart_disease_rate=("has_heart_disease", "mean"),
        )
        .reset_index()
    )
    summary["heart_disease_rate"] *= 100
    return plotted, summary


def risk_tier(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Moderate"
    return "High"


def local_prediction(values: dict) -> dict:
    artifact = load_model_artifact()
    frame = pd.DataFrame([values], columns=artifact["feature_names"])
    pipeline = artifact["pipeline"]
    prediction = int(pipeline.predict(frame)[0])
    positive_index = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(frame)[0][positive_index])
    return {
        "prediction": prediction,
        "label": "Heart disease detected" if prediction else "No heart disease detected",
        "disease_probability": probability,
        "risk_tier": risk_tier(probability),
        "model_name": artifact["model_name"],
    }


def show_prediction(result: dict, source: str) -> None:
    st.subheader(result["label"])
    probability = result["disease_probability"]
    st.metric("Estimated disease probability", f"{probability:.1%}")
    st.progress(probability)
    st.write(f"**Risk tier:** {result['risk_tier']}  ·  **Model:** {result['model_name']}")
    st.caption(f"Prediction served by the {source}.")


st.title("UCI Cleveland Heart Disease Explorer")
st.caption(
    "An educational machine-learning demonstration using the UCI Cleveland dataset."
)
st.warning(
    "Academic and demonstration purposes only. This tool is not a clinical diagnosis "
    "or a substitute for advice from a qualified healthcare professional."
)

with st.sidebar:
    st.header("Prediction service")
    api_url = st.text_input("FastAPI prediction URL", value=DEFAULT_API_URL)
    st.caption("If the API is offline, predictions use the locally trained model.")

predict_tab, eda_tab, clustering_tab = st.tabs(
    ["Risk Predictor", "EDA & Cholesterol Trends", "PCA & Patient Clustering"]
)

with predict_tab:
    st.header("Estimate a model risk score")
    with st.form("risk_prediction_form"):
        col1, col2, col3 = st.columns(3)
        age = col1.slider("Age (years)", 18, 100, 54)
        sex_label = col1.selectbox("Sex (dataset encoding)", ["Female (0)", "Male (1)"])
        chest_pain = col1.selectbox(
            "Chest pain type",
            [
                "Typical angina (1)",
                "Atypical angina (2)",
                "Non-anginal pain (3)",
                "Asymptomatic (4)",
            ],
        )
        resting_bp = col2.slider("Resting blood pressure (mm Hg)", 70, 250, 130)
        cholesterol = col2.slider("Serum cholesterol (mg/dL)", 50, 700, 240)
        fasting_sugar = col2.checkbox("Fasting blood sugar > 120 mg/dL")
        resting_ecg = col2.selectbox(
            "Resting ECG",
            [
                "Normal (0)",
                "ST-T wave abnormality (1)",
                "Left ventricular hypertrophy (2)",
            ],
        )
        max_heart_rate = col3.slider("Maximum heart rate (bpm)", 60, 250, 150)
        exercise_angina = col3.checkbox("Exercise-induced angina")
        oldpeak = col3.slider(
            "ST depression (oldpeak)", -2.0, 10.0, 1.0, step=0.1
        )
        slope = col1.selectbox(
            "Peak exercise ST slope",
            ["Upsloping (1)", "Flat (2)", "Downsloping (3)"],
        )
        ca_option = col2.selectbox(
            "Major vessels colored by fluoroscopy (ca)",
            ["Not provided", "0", "1", "2", "3"],
        )
        thal_option = col3.selectbox(
            "Thalassemia (thal)",
            ["Not provided", "Normal (3)", "Fixed defect (6)", "Reversible defect (7)"],
        )
        submitted = st.form_submit_button("Estimate risk")

    if submitted:
        values = {
            "age": age,
            "sex": int(sex_label.endswith("(1)")),
            "cp": int(chest_pain[-2]),
            "trestbps": resting_bp,
            "chol": cholesterol,
            "fbs": int(fasting_sugar),
            "restecg": int(resting_ecg[-2]),
            "thalach": max_heart_rate,
            "exang": int(exercise_angina),
            "oldpeak": oldpeak,
            "slope": int(slope[-2]),
            "ca": None if ca_option == "Not provided" else float(ca_option),
            "thal": None if thal_option == "Not provided" else float(thal_option[-2]),
        }
        try:
            response = requests.post(api_url, json=values, timeout=3)
        except (requests.ConnectionError, requests.Timeout):
            try:
                result = local_prediction(values)
            except FileNotFoundError as error:
                st.error(str(error))
            else:
                show_prediction(result, "local model")
        else:
            if response.ok:
                show_prediction(response.json(), "FastAPI")
            else:
                st.error(
                    f"FastAPI returned HTTP {response.status_code}: {response.text}"
                )

with eda_tab:
    st.header("Explore the Cleveland dataset")
    X, y = load_dataset()
    left, right = st.columns(2)
    with left:
        st.subheader("Heart disease target distribution")
        class_counts = y.map({0: "No disease", 1: "Disease"}).value_counts()
        st.bar_chart(class_counts)
    with right:
        st.subheader("Feature correlations with cholesterol")
        correlations = X.corr(numeric_only=True)["chol"].drop("chol")
        st.bar_chart(correlations.sort_values())
    selected_feature = st.selectbox(
        "Clinical feature distribution",
        ["age", "trestbps", "chol", "thalach", "oldpeak"],
        format_func=lambda name: {
            "age": "Age (years)",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dL)",
            "thalach": "Maximum heart rate (bpm)",
            "oldpeak": "ST depression",
        }[name],
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=X,
        x=selected_feature,
        hue=y.map({0: "No disease", 1: "Disease"}),
        bins=20,
        ax=ax,
    )
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)
    plt.close(fig)

with clustering_tab:
    st.header("Segment patients by clinical profile")
    X, y = load_dataset()
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=3)
    points, cluster_summary = cluster_patients(X, y, n_clusters)
    selected_cluster = st.selectbox(
        "Highlight cluster",
        ["All clusters"] + [f"Cluster {index}" for index in range(n_clusters)],
    )
    if selected_cluster == "All clusters":
        visible_points = points
    else:
        cluster_id = int(selected_cluster.split()[-1])
        visible_points = points.loc[points["cluster"] == cluster_id]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=visible_points,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="viridis",
        s=55,
        ax=ax,
    )
    ax.set_title("Patient clusters projected onto the first two PCA components")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("PCA retains 95% of feature variance; clustering uses the full reduced space.")
    st.subheader("Cluster profiles")
    st.dataframe(
        cluster_summary.style.format(
            {
                "average_age": "{:.1f}",
                "average_blood_pressure": "{:.1f}",
                "average_cholesterol": "{:.1f}",
                "heart_disease_rate": "{:.1f}%",
            }
        ),
    )
