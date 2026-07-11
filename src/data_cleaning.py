"""
Data Cleaning utilities — Finance Analytics
---------------------------------------------

"""

import numpy as np
import pandas as pd


def load_raw(customers_path: str, transactions_path: str):
    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path, parse_dates=["TransactionDate"])
    return customers, transactions


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2)
    report = pd.DataFrame({"MissingCount": total, "MissingPercent": pct})
    report = report[report["MissingCount"] > 0].sort_values("MissingCount", ascending=False)
    return report.reset_index().rename(columns={"index": "Column"})


def duplicate_report(df: pd.DataFrame, exclude_col: str) -> dict:
    dup_mask = df.duplicated(subset=[c for c in df.columns if c != exclude_col])
    return {"duplicate_rows": int(dup_mask.sum())}


def iqr_outlier_report(df: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        rows.append({
            "Column": col, "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
            "LowerBound": round(lower, 2), "UpperBound": round(upper, 2),
            "OutlierCount": len(outliers),
        })
    return pd.DataFrame(rows)


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median()).round(0).astype(int)
    df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["CustomerName"] = df["CustomerName"].fillna("Unknown Customer")

    # Cap CreditScore to a valid 300-900 range using IQR-informed clipping
    q1, q3 = df["CreditScore"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = max(300, q1 - 1.5 * iqr)
    upper = min(900, q3 + 1.5 * iqr)
    df["CreditScore"] = df["CreditScore"].clip(lower=lower, upper=upper)

    return df


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "TransactionID"])

    df["BalanceAfterTxn"] = df["BalanceAfterTxn"].fillna(df["BalanceAfterTxn"].median())
    df["Merchant"] = df["Merchant"].fillna("Unknown Merchant")
    df["PaymentMode"] = df["PaymentMode"].fillna(df["PaymentMode"].mode()[0])

    # Cap Amount outliers via IQR
    q1, q3 = df["Amount"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["Amount"] = df["Amount"].clip(lower=max(0, lower), upper=upper)

    return df.reset_index(drop=True)


def build_customer_features(customers_clean: pd.DataFrame, transactions_clean: pd.DataFrame) -> pd.DataFrame:
    
    agg = transactions_clean.groupby("CustomerID").agg(
        TxnCount=("TransactionID", "count"),
        TotalDebit=("Amount", lambda s: s[transactions_clean.loc[s.index, "TransactionType"] == "Debit"].sum()),
        TotalCredit=("Amount", lambda s: s[transactions_clean.loc[s.index, "TransactionType"] == "Credit"].sum()),
        AvgBalance=("BalanceAfterTxn", "mean"),
        FraudFlags=("IsFraudulent", lambda s: (s == "Yes").sum()),
    ).reset_index()

    merged = customers_clean.merge(agg, on="CustomerID", how="left")
    merged[["TxnCount", "TotalDebit", "TotalCredit", "AvgBalance", "FraudFlags"]] = (
        merged[["TxnCount", "TotalDebit", "TotalCredit", "AvgBalance", "FraudFlags"]].fillna(0)
    )
    merged["SavingsRate"] = np.where(
        merged["TotalCredit"] > 0,
        ((merged["TotalCredit"] - merged["TotalDebit"]) / merged["TotalCredit"]).clip(-2, 1),
        0,
    )
    return merged
