# Crypto Investment Manager · Portfolio Analytics and Strategy Lab

> A Python and Streamlit toolkit for building, testing, and monitoring crypto portfolios with rule-based allocation, Sharpe-driven weighting, ML-based return forecasting, and automated risk alerts.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Local_DB-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ridge_Model-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-150458?style=flat-square&logo=pandas&logoColor=white)

---

## What is this project?

Crypto Investment Manager is an end-to-end portfolio analysis workspace for cryptocurrency data.

It lets you:

- Upload your own OHLC CSV files (or use local DB-backed data paths).
- Compute technical indicators and trading signals.
- Compare rule-based portfolio weighting approaches.
- Run a Sharpe-based dynamic allocation strategy and stress-test outcomes.
- Forecast returns with a rolling Ridge Regression model.
- Monitor risk thresholds and trigger email alerts when violations occur.

The project ships with a Streamlit interface that ties authentication, analytics, modeling, and risk controls into one workflow.

---

## System Architecture

```text
      ┌────────────────────────────────────────────────────────────────┐
      │                     Streamlit Frontend                         │
      │                                                                │
      │  • Login / Signup                                              │
      │  • CSV Upload                                                  │
      │  • Metrics, Strategy, Risk, Prediction Pages                   │
      └───────────────────────────────┬────────────────────────────────┘
                                      │
                                      │ function calls
                                      ▼
      ┌────────────────────────────────────────────────────────────────┐
      │                     Python Analytics Layer                     │
      │                                                                │
      │  main.py             → technical metrics + signals             │
      │  portfolio_math.py   → Equal / Price / InvVol comparisons      │
      │  investment_rule.py  → Sharpe-based dynamic weighting          │
      │  predictor.py        → rolling Ridge regression forecasts      │
      │  Risk_checker.py     → risk metrics + threshold checks         │
      └───────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
      ┌────────────────────────────────────────────────────────────────┐
      │                       Persistence Layer                        │
      │                                                                │
      │  db/crypto.db  → portfolio history, weights, risk metrics      │
      │  db/user.db    → users and hashed credentials                  │
      └────────────────────────────────────────────────────────────────┘
```

---

## Features

### Streamlit-Based Portfolio Workspace

Single application flow for:

- Authentication (signup/login).
- Data upload and analysis.
- Portfolio strategy comparison.
- Return forecasting dashboards.
- Risk monitoring and alerting.

### Technical Metrics Engine

The metrics pipeline computes:

- Percent change.
- Moving averages.
- Rolling volatility.
- Basic rule-driven trading signals (BUY/SELL/HOLD).

Processing runs concurrently for uploaded assets using thread pools.

### Rule-Based Portfolio Construction

Portfolio comparisons supported:

- Equal weighting.
- Price weighting.
- Inverse volatility weighting.

A weight-cap helper limits overconcentration to improve diversification behavior.

### Sharpe-Driven Dynamic Strategy

The strategy pipeline:

- Merges uploaded assets by date.
- Computes log-return Sharpe ratios.
- Converts normalized Sharpe signals into portfolio weights.
- Produces latest-day portfolio return estimate.
- Stores weight history in SQLite.

### Stress Testing

Monte Carlo scenario simulation for:

- Bull market.
- Bear market.
- Volatile market.

Each scenario returns mean, volatility, and return range to aid decision support.

### Return Prediction (Machine Learning)

Rolling-window Ridge Regression model with feature engineering:

- Inputs: Open, High, Low, Prev_Close, MA3, MA7, MA14.
- Target: next-step return.
- Outputs: train/test scores, MSE, latest actual vs predicted, next-day predicted return.

Predictions are computed for both individual assets and a portfolio-level aggregate.

### Risk Monitoring and Alerts

Risk module computes and stores:

- Volatility (annualized).
- Sharpe ratio.
- Sortino ratio.
- Maximum drawdown.
- Beta proxy.
- Max asset weight.

If thresholds are breached, the module prepares an alert and can send email via SMTP credentials in environment variables.

---

## Key Engineering Choices

### Unified Interactive App

Instead of separate scripts for each analytics stage, the Streamlit app centralizes all workflows into one navigable interface.

### Strategy + Monitoring Coupling

Portfolio construction and risk checks are designed to operate in the same session, reducing gaps between allocation decisions and risk visibility.

### Local-First Persistence

SQLite storage keeps setup lightweight and reproducible for local experimentation and academic usage.

### Dual-Path Data Handling

Most modules can work from either uploaded dataframes or DB tables, allowing quick experimentation and repeatable local runs.

---

## Tech Stack

| Layer            | Technology                                     |
| ---------------- | ---------------------------------------------- |
| Language         | Python 3                                       |
| UI               | Streamlit                                      |
| Data Processing  | Pandas, NumPy                                  |
| Machine Learning | scikit-learn (RidgeCV, preprocessing)          |
| Visualization    | Matplotlib                                     |
| Storage          | SQLite                                         |
| Security         | SHA-256 password hashing                       |
| Alerts           | SMTP email (via python stdlib + dotenv config) |

---

## Project Structure

```text
crypto investment manager/
├── app.py                    # Streamlit entry point
├── auth.py                   # User signup/login + password hashing
├── DB_portfolio.py           # SQLite schema + portfolio persistence
├── main.py                   # Technical metrics and trading signals
├── portfolio_math.py         # Equal/Price/InvVol comparison + plots
├── investment_rule.py        # Sharpe-based weighting + stress tests
├── predictor.py              # Rolling Ridge return forecasting
├── Risk_checker.py           # Risk metrics and alert logic
├── data/                     # Sample market CSVs
├── db/                       # SQLite databases
└── README.md
```

---

## Running Locally

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib python-dotenv
```

### 3) Configure environment variables for risk email alerts (optional)

Create a `.env` file in project root:

```env
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_email_password
ALERT_TO=recipient@example.com
```

### 4) Launch the app

```bash
streamlit run app.py
```

The app initializes database tables automatically on startup.

---

## Typical Workflow

1. Start the app and create/login user credentials.
2. Upload one or more OHLC CSV files (must include date and close price columns; prediction flow needs Open/High/Low/Close).
3. Run technical metric calculation.
4. Compare portfolio strategies (Equal, Price, Inverse Volatility).
5. Run Sharpe-based investment strategy and review stress-test outputs.
6. Execute prediction models for assets and portfolio.
7. Run risk check and review alert status.

---

## Module Reference

| Module             | Responsibility                                                   |
| ------------------ | ---------------------------------------------------------------- |
| app.py             | Streamlit UI orchestration, authentication flow, page navigation |
| auth.py            | User table setup and credential verification                     |
| DB_portfolio.py    | Portfolio table initialization and storage helpers               |
| main.py            | Technical indicators and signal generation functions             |
| portfolio_math.py  | Strategy comparison engine and plotting                          |
| investment_rule.py | Sharpe allocation, scenario simulation, weight persistence       |
| predictor.py       | Rolling Ridge pipeline with engineered features                  |
| Risk_checker.py    | Risk metric computation, threshold checks, optional email alerts |

---

## Notes and Limitations

- This is a research and educational analytics project, not financial advice.
- Model outputs are statistical estimates and should not be used as sole trading signals.
- Uploaded data quality directly impacts strategy and prediction reliability.

---

## License

Licensed under the terms described in the LICENSE file.
