import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ALLOWED_EXTENSIONS = {"csv"}


# ------------------------------------
# Check File Extension
# ------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------
# Data Preprocessing
# ------------------------------------
def preprocess_data(df):

    # Remove Missing Values
    df = df.dropna()

    # Encode Categorical Columns
    label_encoder = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = label_encoder.fit_transform(df[column])

    # Last Column is Target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, X_columns


# ------------------------------------
# Heatmap
# ------------------------------------
def generate_heatmap(df):

    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    path = "static/heatmap.png"

    plt.savefig(path, bbox_inches="tight")

    plt.close()

    return path


# ------------------------------------
# Feature Importance
# ------------------------------------
def plot_feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        return None

    importance = model.feature_importances_

    plt.figure(figsize=(10, 6))

    plt.barh(feature_names, importance)

    plt.xlabel("Importance")

    plt.ylabel("Features")

    plt.title("Feature Importance")

    plt.tight_layout()

    path = "static/feature_importance.png"

    plt.savefig(path)

    plt.close()

    return path


# ------------------------------------
# Dataset Information
# ------------------------------------
def dataset_info(df):

    info = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum(),
        "Column Names": list(df.columns)
    }

    return info


# ------------------------------------
# Class Distribution
# ------------------------------------
def class_distribution(df):

    target = df.columns[-1]

    plt.figure(figsize=(6, 4))

    df[target].value_counts().plot(kind="bar")

    plt.title("Target Distribution")

    plt.xlabel(target)

    plt.ylabel("Count")

    path = "static/class_distribution.png"

    plt.savefig(path)

    plt.close()

    return path