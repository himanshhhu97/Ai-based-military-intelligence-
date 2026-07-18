"""
Trains and saves BOTH machine-learning models used by the dashboard:

  1. Attack Type Prediction  -> models/attack_prediction_model.pkl
  2. Threat Level Prediction -> models/threat_level_model.pkl

Run this once (or whenever the dataset changes) BEFORE launching the app:

    python train_models.py

Both pages then just load the saved model with st.cache_resource instead
of retraining a RandomForest on every single page load, which is what the
original Threat_Level page did (a serious performance bug on a 180k-row
dataset).
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join("data", "globalterrorism.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_raw():
    print("Loading GTD dataset...")
    df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
    print("Raw shape:", df.shape)
    return df


# ------------------------------------------------------------------
# 1. Attack Type Prediction Model
# ------------------------------------------------------------------
def train_attack_type_model(df_raw: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Training Attack Type Prediction Model")
    print("=" * 60)

    features = [
        "country_txt", "region_txt", "weaptype1_txt",
        "targtype1_txt", "gname", "success", "suicide", "nkill", "nwound",
    ]
    target = "attacktype1_txt"

    df = df_raw[features + [target]].copy()
    df["nkill"] = pd.to_numeric(df["nkill"], errors="coerce")
    df["nwound"] = pd.to_numeric(df["nwound"], errors="coerce")
    df = df.dropna()
    print("After cleaning:", df.shape)

    encoders = {}
    for col in ["country_txt", "region_txt", "weaptype1_txt", "targtype1_txt", "gname"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    df[target] = target_encoder.fit_transform(df[target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # NOTE: n_estimators/max_depth are deliberately modest. With
    # high-cardinality categorical features (gname has 3,000+ groups),
    # deep unpruned trees balloon into a 800MB+ pickle for very little
    # accuracy gain. min_samples_leaf caps individual leaf/tree size.
    model = RandomForestClassifier(
        n_estimators=150, max_depth=14, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, pred, target_names=target_encoder.classes_, zero_division=0))

    joblib.dump(model, os.path.join(MODEL_DIR, "attack_prediction_model.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "feature_encoders.pkl"))

    # Feature importance (used by the Attack Prediction page — new addition)
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    joblib.dump(importance, os.path.join(MODEL_DIR, "attack_feature_importance.pkl"))

    print("Saved attack type model, encoders, target encoder, feature importance.")


# ------------------------------------------------------------------
# 2. Threat Level Prediction Model
# ------------------------------------------------------------------
def classify_threat(x):
    if x <= 2:
        return "LOW"
    elif x <= 10:
        return "MEDIUM"
    else:
        return "HIGH"


def train_threat_level_model(df_raw: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Training Threat Level Prediction Model")
    print("=" * 60)

    cols = [
        "country_txt", "region_txt", "attacktype1_txt",
        "weaptype1_txt", "targtype1_txt", "suicide", "nkill", "nwound",
    ]
    df = df_raw[cols].copy()
    df["nkill"] = pd.to_numeric(df["nkill"], errors="coerce")
    df["nwound"] = pd.to_numeric(df["nwound"], errors="coerce")
    df["suicide"] = pd.to_numeric(df["suicide"], errors="coerce")
    df = df.dropna()

    df["impact"] = df["nkill"] + df["nwound"]
    df["threat_level"] = df["impact"].apply(classify_threat)

    encoders = {}
    for col in ["country_txt", "region_txt", "attacktype1_txt", "weaptype1_txt", "targtype1_txt"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    df["threat_level"] = target_encoder.fit_transform(df["threat_level"])

    # NOTE: nkill/nwound are deliberately EXCLUDED from the feature set.
    # They were used above only to construct the label. Including them as
    # inputs (as the original script did) is data leakage -- the model
    # would just be learning to re-derive the label from itself. This
    # version predicts severity from pre-attack context only.
    feature_cols = ["country_txt", "region_txt", "attacktype1_txt", "weaptype1_txt", "targtype1_txt", "suicide"]
    X = df[feature_cols]
    y = df["threat_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, pred, target_names=target_encoder.classes_, zero_division=0))

    joblib.dump(model, os.path.join(MODEL_DIR, "threat_level_model.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "threat_target_encoder.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "threat_feature_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "threat_feature_cols.pkl"))

    print("Saved threat level model, encoders, target encoder.")


if __name__ == "__main__":
    df_raw = load_raw()
    train_attack_type_model(df_raw)
    train_threat_level_model(df_raw)
    print("\nAll models trained and saved to ./models")
