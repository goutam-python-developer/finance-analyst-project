"""
Machine Learning models — Finance Analytics project.

  - Linear Regression      -> predict CreditScore from financial behaviour
  - Logistic Regression /
    Random Forest          -> predict Loan Default risk
  - K-Means Clustering     -> customer segmentation by financial behaviour
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score,
)


def train_credit_score_regression(features_df: pd.DataFrame) -> dict:

    cols = ["MonthlyIncome", "TxnCount", "TotalDebit", "TotalCredit", "AvgBalance", "SavingsRate"]
    data = features_df.dropna(subset=cols + ["CreditScore"])
    X = data[cols]
    y = data["CreditScore"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model,
        "features": cols,
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "r2": round(r2_score(y_test, preds), 4),
        "coefficients": dict(zip(cols, np.round(model.coef_, 4))),
        "intercept": round(model.intercept_, 3),
    }


def train_loan_default_classifier(features_df: pd.DataFrame) -> dict:
    """Logistic Regression + Random Forest: predict LoanDefault."""
    data = features_df[features_df["LoanTenureMonths"] > 0].copy()
    le_city = LabelEncoder()
    le_acc = LabelEncoder()
    data["CityEnc"] = le_city.fit_transform(data["City"])
    data["AccEnc"] = le_acc.fit_transform(data["AccountType"])
    data["DefaultFlag"] = (data["LoanDefault"] == "Yes").astype(int)

    cols = ["Age", "CreditScore", "MonthlyIncome", "LoanAmount", "LoanTenureMonths",
            "TxnCount", "AvgBalance", "SavingsRate", "CityEnc", "AccEnc"]
    X = data[cols]
    y = data["DefaultFlag"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    log_preds = log_reg.predict(X_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        }

    feature_importance = dict(zip(cols, np.round(rf.feature_importances_, 3)))

    return {
        "logistic_regression": metrics(y_test, log_preds),
        "random_forest": metrics(y_test, rf_preds),
        "feature_importance": feature_importance,
        "features": cols,
        "default_rate": round(y.mean() * 100, 2),
    }


def customer_risk_segmentation(features_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """K-Means clustering on Income, Savings Rate & Credit Score -> risk/value segments."""
    df = features_df.copy()
    cols = ["MonthlyIncome", "SavingsRate", "CreditScore", "TotalDebit"]
    df = df.dropna(subset=cols)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Segment"] = km.fit_predict(X)

    seg_order = df.groupby("Segment")["CreditScore"].mean().sort_values(ascending=False).index
    labels = ["Prime / Low Risk", "Stable Savers", "Growing Spenders", "High Risk"]
    label_map = {seg: labels[i] if i < len(labels) else f"Segment {i}" for i, seg in enumerate(seg_order)}
    df["SegmentLabel"] = df["Segment"].map(label_map)

    return df
