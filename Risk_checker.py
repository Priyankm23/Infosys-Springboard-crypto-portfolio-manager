import sqlite3
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# ==== CONFIG ====
DB_PATH = "db/crypto.db"
THRESHOLDS = {
    "volatility": 0.05,      # < 5%
    "sharpe": 1.0,           # ≥ 1
    "max_drawdown": -0.20,   # ≥ -20%
    "sortino": 1.0,          # ≥ 1
    "beta": 1.2,             # < 1.2
    "max_weight": 0.5        # < 50%
}

# Email setup
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
ALERT_TO = os.getenv("ALERT_TO")

def fetch_data(uploaded_data=None):
    """Fetch Close prices from all asset tables or uploaded data."""
    if uploaded_data is not None:
        # Uploaded data assumed to be a dict {symbol: DataFrame}
        data = {}
        for symbol, df in uploaded_data.items():
            df.columns = [col.strip().title() for col in df.columns]
            if 'Close' in df.columns and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date', 'Close'])
                df.set_index("Date", inplace=True)
                data[symbol] = pd.to_numeric(df['Close'], errors='coerce')
        prices = pd.concat(data, axis=1)
        prices = prices.dropna(how="any")
        return prices
    else:
        # Default DB fetch
        conn = sqlite3.connect(DB_PATH)
        ASSETS = ["btc_prices", "eth_prices", "usdc_prices"]
        data = {}
        for asset in ASSETS:
            query = f"SELECT Date, Close FROM {asset} ORDER BY Date ASC"
            df = pd.read_sql_query(query, conn, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            data[asset] = df["Close"]
        conn.close()
        prices = pd.concat(data, axis=1)
        prices = prices.dropna(how="any")
        return prices

def compute_metrics(prices):
    """Compute all risk metrics for equal-weight portfolio"""
    returns = prices.pct_change(fill_method=None).dropna()
    n_assets = returns.shape[1]
    weights = np.repeat(1 / n_assets, n_assets) if n_assets > 0 else []

    port_ret = returns.dot(weights)
    vol = port_ret.std() * np.sqrt(252) if not port_ret.empty else np.nan

    sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() != 0 else np.nan
    downside = port_ret[port_ret < 0].std()
    sortino = (port_ret.mean() / downside) * np.sqrt(252) if downside != 0 else np.nan

    cumulative = (1 + port_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    mdd = drawdown.min() if not drawdown.empty else np.nan

    if not returns.empty:
        market = returns.iloc[:, 0]  # Take first asset as benchmark
        cov = np.cov(port_ret, market)[0][1]
        var = np.var(market)
        beta = cov / var if var != 0 else np.nan
    else:
        beta = np.nan

    max_weight = weights.max() if weights.size > 0 else np.nan

    return {
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "beta": beta,
        "max_weight": max_weight
    }

def store_metrics(metrics):
    """Store results in DB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_metrics (
            date TEXT,
            volatility REAL,
            sharpe REAL,
            sortino REAL,
            max_drawdown REAL,
            beta REAL,
            max_weight REAL
        )
    """)
    conn.commit()
    cursor.execute(
        "INSERT INTO risk_metrics (date, volatility, sharpe, sortino, max_drawdown, beta, max_weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics["volatility"], metrics["sharpe"], metrics["sortino"],
            metrics["max_drawdown"], metrics["beta"], metrics["max_weight"]
        )
    )
    conn.commit()
    conn.close()

def check_and_prepare_alert(metrics):
    """Checks for threshold violations and prepares an alert message."""
    violations = []

    if metrics["volatility"] >= THRESHOLDS["volatility"]:
        violations.append(f"Volatility {metrics['volatility']:.2%} ≥ {THRESHOLDS['volatility']:.0%}")
    if metrics["sharpe"] < THRESHOLDS["sharpe"]:
        violations.append(f"Sharpe {metrics['sharpe']:.2f} < {THRESHOLDS['sharpe']}")
    if metrics["sortino"] < THRESHOLDS["sortino"]:
        violations.append(f"Sortino {metrics['sortino']:.2f} < {THRESHOLDS['sortino']}")
    if metrics["max_drawdown"] < THRESHOLDS["max_drawdown"]:
        violations.append(f"Max Drawdown {metrics['max_drawdown']:.2%} < {THRESHOLDS['max_drawdown']:.0%}")
    if metrics["beta"] is not None and metrics["beta"] >= THRESHOLDS["beta"]:
        violations.append(f"Beta {metrics['beta']:.2f} ≥ {THRESHOLDS['beta']:.2f}")
    if metrics["max_weight"] >= THRESHOLDS["max_weight"]:
        violations.append(f"Max Weight {metrics['max_weight']:.2%} ≥ {THRESHOLDS['max_weight']:.0%}")

    if not violations:
        return None

    return f"""⚠️ Risk Alert Triggered ⚠️

Violations:
- {"\n- ".join(violations)}

Full Metrics:
Volatility: {metrics['volatility']:.2%}
Sharpe Ratio: {metrics['sharpe']:.2f}
Sortino Ratio: {metrics['sortino']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Beta (vs BTC): {metrics['beta']:.2f}
Max Asset Weight: {metrics['max_weight']:.2%}
"""

def run_risk_check(uploaded_data=None):
    """
    Fetches data (from uploaded CSV or DB), computes risk metrics,
    stores them, and returns metrics and any alert message.
    """
    prices = fetch_data(uploaded_data)  # pass uploaded_data
    metrics = compute_metrics(prices)
    store_metrics(metrics)
    alert_message = check_and_prepare_alert(metrics)

    # Send email if there is a violation
    if alert_message:
        try:
            email = MIMEText(alert_message)
            email["Subject"] = "⚠ Crypto Risk Alert ⚠"
            email["From"] = EMAIL_USER
            email["To"] = ALERT_TO

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(email)

            print("\nRisk alert email sent successfully.")
        except Exception as e:
            print(f"\nFailed to send risk alert email: {e}")

    return metrics, alert_message


if __name__ == "__main__":
    metrics, alert_message = run_risk_check()
    print(metrics)
    if alert_message:
        print(alert_message)
