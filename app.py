"""
FinSight — Finance Analytics Dashboard
=========================================

"""

import io
import json
import os

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify

from src.data_cleaning import (
    load_raw, missing_value_report, duplicate_report,
    iqr_outlier_report, clean_customers, clean_transactions, build_customer_features,
)
from src.ml_models import (
    train_credit_score_regression, train_loan_default_classifier, customer_risk_segmentation,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMERS_PATH = os.path.join(BASE_DIR, "data", "customers.csv")
TRANSACTIONS_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")

app = Flask(__name__)

# ---- Load & prepare data once at startup ----
RAW_CUSTOMERS, RAW_TXNS = load_raw(CUSTOMERS_PATH, TRANSACTIONS_PATH)
CLEAN_CUSTOMERS = clean_customers(RAW_CUSTOMERS)
CLEAN_TXNS = clean_transactions(RAW_TXNS)
CUSTOMER_FEATURES = build_customer_features(CLEAN_CUSTOMERS, CLEAN_TXNS)


def fig_json(fig_dict):
    return json.dumps(fig_dict)


def base_layout(title, height=320, legend=False):
    return {
        "title": {"text": title, "font": {"family": "Space Grotesk, sans-serif", "size": 14, "color": "#0F3D2E"}},
        "height": height,
        "margin": {"l": 50, "r": 20, "t": 45, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "size": 12, "color": "#33533F"},
        "showlegend": legend,
        "xaxis": {"gridcolor": "#EFEBDD"},
        "yaxis": {"gridcolor": "#EFEBDD"},
    }


def sidebar_ctx():
    return {"total_customers": f"{len(RAW_CUSTOMERS):,}", "total_txns": f"{len(RAW_TXNS):,}"}


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
@app.route("/")
def dashboard():
    txns = CLEAN_TXNS

    total_credit = txns.loc[txns.TransactionType == "Credit", "Amount"].sum()
    total_debit = txns.loc[txns.TransactionType == "Debit", "Amount"].sum()
    net_flow = total_credit - total_debit
    fraud_rate = (txns["IsFraudulent"] == "Yes").mean() * 100

    monthly = txns.copy()
    monthly["Month"] = monthly["TransactionDate"].dt.to_period("M").astype(str)
    monthly_flow = monthly.groupby(["Month", "TransactionType"])["Amount"].sum().reset_index()
    credit_m = monthly_flow[monthly_flow.TransactionType == "Credit"]
    debit_m = monthly_flow[monthly_flow.TransactionType == "Debit"]

    spend_by_cat = txns[txns.TransactionType == "Debit"].groupby("Category")["Amount"].sum() \
        .sort_values(ascending=False).reset_index()

    payment_mix = txns.groupby("PaymentMode")["TransactionID"].count().reset_index()

    city_flow = CLEAN_CUSTOMERS.groupby("City")["MonthlyIncome"].mean().sort_values(ascending=False).head(8).reset_index()

    charts = {
        "monthly_flow": {
            "data": [
                {"x": credit_m["Month"].tolist(), "y": credit_m["Amount"].round(2).tolist(),
                 "type": "scatter", "mode": "lines+markers", "name": "Credit (Income)",
                 "line": {"color": "#1B6E8C", "width": 3}},
                {"x": debit_m["Month"].tolist(), "y": debit_m["Amount"].round(2).tolist(),
                 "type": "scatter", "mode": "lines+markers", "name": "Debit (Expense)",
                 "line": {"color": "#C9A227", "width": 3}},
            ],
            "layout": base_layout("Monthly Cash Flow: Credit vs Debit (₹)", height=320, legend=True),
        },
        "spend_by_category": {
            "data": [{"x": spend_by_cat["Category"].tolist(), "y": spend_by_cat["Amount"].round(2).tolist(),
                       "type": "bar", "marker": {"color": "#0F3D2E"}}],
            "layout": base_layout("Debit Spend by Category (₹)", height=320),
        },
        "payment_mix": {
            "data": [{"labels": payment_mix["PaymentMode"].tolist(), "values": payment_mix["TransactionID"].tolist(),
                       "type": "pie", "hole": 0.55,
                       "marker": {"colors": ["#0F3D2E", "#C9A227", "#1B6E8C", "#B3413B", "#8C7AE6", "#6D8A78"]}}],
            "layout": base_layout("Transactions by Payment Mode", height=320, legend=True),
        },
        "avg_income_by_city": {
            "data": [{"x": city_flow["MonthlyIncome"].round(0).tolist(), "y": city_flow["City"].tolist(),
                       "type": "bar", "orientation": "h", "marker": {"color": "#1B6E8C"}}],
            "layout": base_layout("Avg Monthly Income — Top 8 Cities (₹)", height=320),
        },
    }

    ctx = sidebar_ctx()
    return render_template(
        "index.html", active="dashboard", **ctx,
        total_credit=f"{total_credit:,.0f}", total_debit=f"{total_debit:,.0f}",
        net_flow=f"{net_flow:,.0f}", fraud_rate=f"{fraud_rate:.2f}",
        charts_json=fig_json(charts),
    )


# ----------------------------------------------------------------------
# Data Cleaning
# ----------------------------------------------------------------------
@app.route("/cleaning")
def cleaning():
    miss_cust = missing_value_report(RAW_CUSTOMERS)
    miss_txn = missing_value_report(RAW_TXNS)
    dupes = duplicate_report(RAW_TXNS, "TransactionID")
    outliers = iqr_outlier_report(RAW_TXNS, ["Amount"])
    outliers_cust = iqr_outlier_report(RAW_CUSTOMERS.dropna(subset=["CreditScore"]), ["CreditScore", "LoanAmount"])

    ctx = sidebar_ctx()
    return render_template(
        "cleaning.html", active="cleaning", **ctx,
        raw_customers=len(RAW_CUSTOMERS), raw_txns=len(RAW_TXNS),
        clean_customers=len(CLEAN_CUSTOMERS), clean_txns=len(CLEAN_TXNS),
        missing_cust=miss_cust.to_dict("records"), missing_txn=miss_txn.to_dict("records"),
        dup_count=dupes["duplicate_rows"],
        outliers=outliers.to_dict("records") + outliers_cust.to_dict("records"),
    )


# ----------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------
@app.route("/eda")
def eda():
    txns = CLEAN_TXNS
    cust = CLEAN_CUSTOMERS

    income_hist = cust["MonthlyIncome"].tolist()
    credit_hist = cust["CreditScore"].tolist()

    scatter_sample = CUSTOMER_FEATURES.sample(min(400, len(CUSTOMER_FEATURES)), random_state=1)

    corr_cols = ["Age", "MonthlyIncome", "CreditScore", "LoanAmount", "TxnCount",
                 "TotalDebit", "TotalCredit", "AvgBalance", "SavingsRate"]
    corr = CUSTOMER_FEATURES[corr_cols].corr().round(2)

    box_by_type = [
        {"y": txns[txns.TransactionType == t]["Amount"].tolist(), "type": "box", "name": t}
        for t in txns["TransactionType"].unique()
    ]

    charts = {
        "income_hist": {
            "data": [{"x": income_hist, "type": "histogram", "marker": {"color": "#0F3D2E"}, "nbinsx": 20}],
            "layout": base_layout("Univariate: Monthly Income Distribution", height=300),
        },
        "credit_hist": {
            "data": [{"x": credit_hist, "type": "histogram", "marker": {"color": "#C9A227"}, "nbinsx": 20}],
            "layout": base_layout("Univariate: Credit Score Distribution", height=300),
        },
        "income_vs_credit": {
            "data": [{"x": scatter_sample["MonthlyIncome"].tolist(), "y": scatter_sample["CreditScore"].tolist(),
                       "mode": "markers", "type": "scatter",
                       "marker": {"color": "#1B6E8C", "size": 6, "opacity": 0.6}}],
            "layout": base_layout("Bivariate: Monthly Income vs Credit Score", height=320),
        },
        "amount_box": {
            "data": box_by_type,
            "layout": base_layout("Bivariate: Transaction Amount by Type", height=340, legend=False),
        },
        "corr_heatmap": {
            "data": [{"z": corr.values.tolist(), "x": corr.columns.tolist(), "y": corr.columns.tolist(),
                       "type": "heatmap", "colorscale": [[0, "#B3413B"], [0.5, "#F6F4EC"], [1, "#1B6E8C"]],
                       "zmin": -1, "zmax": 1}],
            "layout": base_layout("Multivariate: Correlation Heatmap (Customer Features)", height=440),
        },
    }

    ctx = sidebar_ctx()
    return render_template("eda.html", active="eda", **ctx, charts_json=fig_json(charts))


# ----------------------------------------------------------------------
# ML Models
# ----------------------------------------------------------------------
@app.route("/ml")
def ml():
    reg = train_credit_score_regression(CUSTOMER_FEATURES)
    clf = train_loan_default_classifier(CUSTOMER_FEATURES)
    seg = customer_risk_segmentation(CUSTOMER_FEATURES)

    seg_summary = seg.groupby("SegmentLabel").agg(
        Customers=("CustomerID", "count"),
        AvgIncome=("MonthlyIncome", "mean"),
        AvgCreditScore=("CreditScore", "mean"),
        AvgSavingsRate=("SavingsRate", "mean"),
    ).round(2).reset_index().to_dict("records")

    seg_scatter = {
        "data": [
            {"x": seg[seg.SegmentLabel == s]["MonthlyIncome"].tolist(),
             "y": seg[seg.SegmentLabel == s]["CreditScore"].tolist(),
             "mode": "markers", "type": "scatter", "name": s, "marker": {"size": 8, "opacity": 0.75}}
            for s in seg["SegmentLabel"].unique()
        ],
        "layout": base_layout("Customer Segments — Income vs Credit Score", height=380, legend=True),
    }

    fi = clf["feature_importance"]
    feat_importance_chart = {
        "data": [{"x": list(fi.values()), "y": list(fi.keys()), "type": "bar", "orientation": "h",
                   "marker": {"color": "#0F3D2E"}}],
        "layout": base_layout("Random Forest — Feature Importance (Loan Default)", height=340),
    }

    ctx = sidebar_ctx()
    return render_template(
        "ml.html", active="ml", **ctx,
        reg=reg, clf=clf, seg_summary=seg_summary,
        seg_scatter=fig_json(seg_scatter), feat_importance_chart=fig_json(feat_importance_chart),
    )


@app.route("/ml/predict_credit_score", methods=["POST"])
def predict_credit_score():
    reg = train_credit_score_regression(CUSTOMER_FEATURES)
    model = reg["model"]
    payload = request.get_json()
    x = pd.DataFrame([[float(payload.get(f, 0)) for f in reg["features"]]], columns=reg["features"])
    pred = model.predict(x)[0]
    pred = float(np.clip(pred, 300, 900))
    return jsonify({"predicted_credit_score": round(pred, 0)})


# ----------------------------------------------------------------------
# Browse transactions
# ----------------------------------------------------------------------
@app.route("/browse")
def browse():
    df = CLEAN_TXNS.merge(CLEAN_CUSTOMERS[["CustomerID", "CustomerName", "City"]], on="CustomerID", how="left")

    category = request.args.get("category", "")
    txn_type = request.args.get("type", "")
    payment = request.args.get("payment", "")
    fraud = request.args.get("fraud", "")

    if category:
        df = df[df["Category"] == category]
    if txn_type:
        df = df[df["TransactionType"] == txn_type]
    if payment:
        df = df[df["PaymentMode"] == payment]
    if fraud:
        df = df[df["IsFraudulent"] == fraud]

    df_display = df.sort_values("TransactionDate", ascending=False).head(300).copy()
    df_display["TransactionDate"] = df_display["TransactionDate"].dt.strftime("%Y-%m-%d")

    ctx = sidebar_ctx()
    return render_template(
        "data.html", active="browse", **ctx,
        rows=df_display.to_dict("records"), row_count=len(df),
        categories=sorted(CLEAN_TXNS["Category"].unique()),
        payments=sorted(CLEAN_TXNS["PaymentMode"].unique()),
        selected={"category": category, "type": txn_type, "payment": payment, "fraud": fraud},
    )


# ----------------------------------------------------------------------
# Export cleaned data to Excel
# ----------------------------------------------------------------------
@app.route("/export/excel")
def export_excel():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        CLEAN_CUSTOMERS.to_excel(writer, sheet_name="Cleaned Customers", index=False)
        CLEAN_TXNS.to_excel(writer, sheet_name="Cleaned Transactions", index=False)
        CUSTOMER_FEATURES.to_excel(writer, sheet_name="Customer Features (RFM)", index=False)

        cat_summary = CLEAN_TXNS[CLEAN_TXNS.TransactionType == "Debit"].groupby("Category").agg(
            TxnCount=("TransactionID", "count"), TotalAmount=("Amount", "sum"),
        ).round(2).sort_values("TotalAmount", ascending=False).reset_index()
        cat_summary.to_excel(writer, sheet_name="Category Summary", index=False)

    buf.seek(0)
    return send_file(
        buf, as_attachment=True, download_name="finance_cleaned_data.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    app.run(debug=True)
