"""
Finance Analytics Data Generator
----------------------------------

"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(7)

N_CUSTOMERS = 600
N_TRANSACTIONS = 8000

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
          "Pune", "Ahmedabad", "Ludhiana", "Jaipur", "Lucknow", "Chandigarh"]

ACCOUNT_TYPES = ["Savings", "Current"]

CATEGORIES_DEBIT = ["Rent", "Groceries", "Entertainment", "EMI - Loan", "Utilities",
                     "Healthcare", "Shopping", "Travel", "Insurance Premium", "Dining"]
CATEGORY_CREDIT = ["Salary", "Investment Return", "Refund", "Interest Credit", "Freelance Income"]

PAYMENT_MODES = ["UPI", "Debit Card", "Credit Card", "Net Banking", "Cash", "Cheque"]

MERCHANTS = ["Amazon", "Swiggy", "Zomato", "BigBasket", "Reliance Digital", "IRCTC",
             "LIC", "HDFC Bank", "Netflix", "Local Store", "Uber", "Apollo Pharmacy",
             "Employer Payroll", "Mutual Fund SIP", "Electricity Board"]

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Ananya", "Diya",
               "Isha", "Kavya", "Meera", "Riya", "Karan", "Rohan", "Neha", "Priya",
               "Aman", "Simran", "Harpreet", "Gurpreet", "Nikita", "Rahul", "Sanya", "Yash"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Reddy", "Nair",
              "Iyer", "Kaur", "Malhotra", "Chopra", "Mehta", "Joshi", "Rao", "Bose"]


def random_dates(start, end, n):
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9
    return pd.to_datetime(RNG.integers(start_u, end_u, n), unit="s")


def generate_customers() -> pd.DataFrame:
    customer_ids = [f"CUST{2000 + i}" for i in range(N_CUSTOMERS)]
    names = [f"{RNG.choice(FIRST_NAMES)} {RNG.choice(LAST_NAMES)}" for _ in range(N_CUSTOMERS)]
    ages = RNG.integers(21, 62, N_CUSTOMERS).astype(float)
    genders = RNG.choice(["Male", "Female", "Other"], N_CUSTOMERS, p=[0.49, 0.49, 0.02])
    cities = RNG.choice(CITIES, N_CUSTOMERS)
    account_type = RNG.choice(ACCOUNT_TYPES, N_CUSTOMERS, p=[0.72, 0.28])

    credit_score = RNG.normal(680, 85, N_CUSTOMERS)
    credit_score = np.clip(credit_score, 300, 900)
    # inject a few extreme/glitched scores
    idx = RNG.choice(N_CUSTOMERS, 8, replace=False)
    credit_score[idx] = RNG.uniform(150, 250, 8)

    loan_amount = np.round(np.clip(RNG.exponential(180000, N_CUSTOMERS), 0, 2_000_000), 0)
    loan_tenure = RNG.choice([0, 12, 24, 36, 48, 60, 84, 120], N_CUSTOMERS,
                              p=[0.25, 0.1, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07])

    # Loan default probability increases with low credit score & high loan amount
    default_prob = 1 / (1 + np.exp((credit_score - 620) / 40 - (loan_amount / 500000)))
    default_prob = np.where(loan_tenure == 0, 0.0, default_prob)
    loan_default = RNG.binomial(1, np.clip(default_prob, 0, 0.85))

    monthly_income = np.round(np.clip(RNG.normal(65000, 30000, N_CUSTOMERS), 15000, None), -2)

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "CustomerName": names,
        "Age": ages,
        "Gender": genders,
        "City": cities,
        "AccountType": account_type,
        "MonthlyIncome": monthly_income,
        "CreditScore": np.round(credit_score, 0),
        "LoanAmount": loan_amount,
        "LoanTenureMonths": loan_tenure,
        "LoanDefault": np.where(loan_default == 1, "Yes", "No"),
    })

    # inject missing values on purpose
    for col, frac in [("Age", 0.05), ("CreditScore", 0.06), ("MonthlyIncome", 0.04), ("CustomerName", 0.01)]:
        idx = RNG.choice(N_CUSTOMERS, int(N_CUSTOMERS * frac), replace=False)
        df.loc[idx, col] = np.nan

    return df


def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    txn_ids = [f"TXN{500000 + i}" for i in range(N_TRANSACTIONS)]
    customer_ids = RNG.choice(customers["CustomerID"], N_TRANSACTIONS)
    dates = random_dates(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"), N_TRANSACTIONS)

    txn_type = RNG.choice(["Debit", "Credit"], N_TRANSACTIONS, p=[0.78, 0.22])

    categories = []
    for t in txn_type:
        categories.append(RNG.choice(CATEGORY_CREDIT) if t == "Credit" else RNG.choice(CATEGORIES_DEBIT))

    amount = np.where(
        txn_type == "Credit",
        RNG.normal(45000, 20000, N_TRANSACTIONS),
        RNG.exponential(3200, N_TRANSACTIONS),
    )
    amount = np.clip(amount, 50, None)
    # outliers: a few huge glitch transactions
    idx = RNG.choice(N_TRANSACTIONS, 30, replace=False)
    amount[idx] *= RNG.uniform(8, 20, 30)

    balance_after = np.round(RNG.uniform(500, 500000, N_TRANSACTIONS), 2)
    payment_mode = RNG.choice(PAYMENT_MODES, N_TRANSACTIONS, p=[0.34, 0.22, 0.14, 0.14, 0.1, 0.06])
    merchant = RNG.choice(MERCHANTS, N_TRANSACTIONS)

    fraud_prob = np.where(
        (amount > np.percentile(amount, 97)) & (payment_mode == "Cheque"), 0.35, 0.006
    )
    is_fraud = RNG.binomial(1, fraud_prob)

    df = pd.DataFrame({
        "TransactionID": txn_ids,
        "TransactionDate": dates,
        "CustomerID": customer_ids,
        "TransactionType": txn_type,
        "Category": categories,
        "Amount": np.round(amount, 2),
        "BalanceAfterTxn": balance_after,
        "PaymentMode": payment_mode,
        "Merchant": merchant,
        "IsFraudulent": np.where(is_fraud == 1, "Yes", "No"),
    })

    # missing values on purpose
    for col, frac in [("BalanceAfterTxn", 0.05), ("Merchant", 0.02), ("PaymentMode", 0.02)]:
        idx = RNG.choice(N_TRANSACTIONS, int(N_TRANSACTIONS * frac), replace=False)
        df.loc[idx, col] = np.nan

    # duplicate rows on purpose
    dup_rows = df.sample(20, random_state=2)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sort_values("TransactionDate").reset_index(drop=True)
    return df


if __name__ == "__main__":
    customers = generate_customers()
    transactions = generate_transactions(customers)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    customers.to_csv(os.path.join(out_dir, "customers.csv"), index=False)
    transactions.to_csv(os.path.join(out_dir, "transactions.csv"), index=False)
    print(f"Generated {len(customers)} customers and {len(transactions)} transactions")
