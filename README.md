# FinSight — Finance Analytics Project

A full, working data analytics web application built for a **Finance /
Data Analyst internship submission**. It takes two related (synthetic but
realistic) finance tables — customers and transactions — through data
cleaning → EDA → dashboards → machine learning, served as a Flask website.

🔗 **GitHub repo:** add your repo link here after pushing
🔗 **Deployed link:** add your live Render URL here after deploying (see below)

---

## Why this project

Built to demonstrate hands-on skill in **Python, NumPy, Pandas, statistics,
data cleaning, Excel, and Machine Learning** applied to a realistic finance
business problem: credit scoring, loan default risk, and customer
segmentation — instead of a toy dataset.

| Topic | Where it's used |
|---|---|
| Python, NumPy, Pandas | `src/data_generator.py`, `src/data_cleaning.py` |
| Merge / groupby across two tables | `src/data_cleaning.py: build_customer_features()` |
| Missing values & imputation | `/cleaning` page |
| Outlier handling via IQR | `/cleaning` page |
| EDA (uni/bi/multivariate) | `/eda` page |
| Charting (bar, pie, line, box, heatmap) | Plotly charts on every page |
| Excel export | `/export/excel` — 4-sheet cleaned workbook |
| Dashboard design (KPIs, filters) | `/` dashboard, `/browse` filters |
| Linear Regression | `/ml` — Credit Score prediction, with a live predictor |
| Logistic Regression / Random Forest | `/ml` — Loan default risk classification |
| K-Means clustering | `/ml` — Customer risk/value segmentation |

---

## Features

- **Dashboard** — cash flow KPIs (credit/debit/net flow), monthly trend,
  spend by category, payment mode mix, income by city.
- **Data Cleaning** — transparent missing-value, duplicate, and IQR outlier
  reports for both tables.
- **EDA & Insights** — income/credit-score distributions, income vs credit
  score scatter, transaction amount box plots, correlation heatmap.
- **ML Models** — credit score regression (with a live "try it" form),
  loan default classification (Logistic Regression vs Random Forest,
  compared side by side), and K-Means customer risk segmentation.
- **Browse Transactions** — filterable transaction table (category / type /
  payment mode / fraud flag).
- **Excel export** — one-click, 4-sheet cleaned workbook download.

---

## Tech stack

`Python` · `Flask` · `Pandas` · `NumPy` · `scikit-learn` · `Plotly.js` ·
`openpyxl` · `HTML/CSS/Jinja2` · `Gunicorn` (deployment)

---

## Project structure

```
finance-analyst-project/
├── app.py
├── requirements.txt
├── Procfile                 # for Render/Heroku-style deployment
├── data/
│   ├── customers.csv
│   └── transactions.csv
├── src/
│   ├── data_generator.py
│   ├── data_cleaning.py
│   └── ml_models.py
├── templates/
│   ├── base.html, index.html, cleaning.html, eda.html, ml.html, data.html
└── static/css/style.css
```

---

## Run it locally

```bash
git clone https://github.com/<your-username>/finance-analyst-project.git
cd finance-analyst-project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**.

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Finance analytics project (FinSight)"
git branch -M main
git remote add origin https://github.com/<your-username>/finance-analyst-project.git
git push -u origin main
```

---

## Deploy for free (Render.com) — for the "Deployed Project Link" field

1. Push the project to GitHub (above).
2. Go to **render.com** → sign up/login with GitHub.
3. **New +** → **Web Service** → select this repo.
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance type:** Free
5. Click **Create Web Service** — Render builds and deploys automatically.
6. After a few minutes you'll get a live URL like:
   `https://finance-analyst-project.onrender.com`

That URL is what goes in the **Deployed project link** field.

> Free-tier Render apps sleep after inactivity and take ~30-60s to wake up
> on the first request — this is normal, not a bug.

---

## Author

Built as a project for a Finance / Data Analyst internship application.
