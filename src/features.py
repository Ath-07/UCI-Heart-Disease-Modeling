"""Shared feature definitions for training, serving, and the dashboard."""

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years.",
    "sex": "Biological sex as encoded by the Cleveland dataset (1 = male, 0 = female).",
    "cp": "Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic).",
    "trestbps": "Resting blood pressure in mm Hg on hospital admission.",
    "chol": "Serum cholesterol in mg/dL.",
    "fbs": "Fasting blood sugar > 120 mg/dL (1 = true, 0 = false).",
    "restecg": "Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy).",
    "thalach": "Maximum heart rate achieved in beats per minute.",
    "exang": "Exercise-induced angina (1 = yes, 0 = no).",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping).",
    "ca": "Number of major vessels (0–3) colored by fluoroscopy; may be omitted.",
    "thal": "Thalassemia result (3 = normal, 6 = fixed defect, 7 = reversible defect); may be omitted.",
}
