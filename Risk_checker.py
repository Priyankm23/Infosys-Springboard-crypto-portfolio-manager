import sqlite3
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ==== CONFIG ====
DB_PATH = "db/crypto.db"
ASSETS = ["btc_prices", "eth_prices", "usdc_prices"]
THRESHOLDS = {
    "volatility": 0.05,      # < 5%
    "sharpe": 1.0,           # ≥ 1
    "max_drawdown": -0.20,   # ≥ -20%
    "sortino": 1.0,          # ≥ 1
    "beta": 1.2,             # < 1.2
    "max_weight": 0.5        # < 50%
}

# Email setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "priyankmoradia34@gmail.com"      # change
EMAIL_PASS = "jqrbhwkwdvztjhuz"                # change (use app password)
ALERT_TO = "2023002327.gcet@cvmu.edu.in"       # change

def fetch_data():
    """Fetch Close prices from all asset tables and combine into DataFrame"""
    conn = sqlite3.connect(DB_PATH)
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
    """Compute all 6 risk metrics for equal-weight portfolio"""
    returns = prices.pct_change(fill_method=None).dropna()
    n_assets = len(ASSETS)
    weights = np.repeat(1 / n_assets, n_assets)

    port_ret = returns.dot(weights)
    vol = port_ret.std() * np.sqrt(252)

    sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() != 0 else 0
    downside = port_ret[port_ret < 0].std()
    sortino = (port_ret.mean() / downside) * np.sqrt(252) if downside != 0 else 0

    cumulative = (1 + port_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    mdd = drawdown.min()

    market = returns["btc_prices"]
    cov = np.cov(port_ret, market)[0][1]
    var = np.var(market)
    beta = cov / var if var != 0 else np.nan

    max_weight = weights.max()

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
        violations.append(f"Beta {metrics['beta']:.2f} ≥ {THRESHOLDS['beta']}")
    if metrics["max_weight"] >= THRESHOLDS["max_weight"]:
        violations.append(f"Max Weight {metrics['max_weight']:.2%} ≥ {THRESHOLDS['max_weight']:.0%}")

    if not violations:
        return None

    msg = f"""⚠️ Risk Alert Triggered ⚠️

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
    return msg

def run_risk_check():
    """Fetches data, computes metrics, stores them, and returns metrics and alerts."""
    prices = fetch_data()
    metrics = compute_metrics(prices)
    store_metrics(metrics)
    alert_message = check_and_prepare_alert(metrics)
    return metrics, alert_message

if __name__ == "__main__":
    metrics, alert_message = run_risk_check()

    print("\n=== Portfolio Risk Metrics ===")
    for k, v in metrics.items():
        if k in ["volatility", "max_drawdown", "max_weight"]:
            print(f"{k.capitalize()}: {v:.2%}")
        else:
            print(f"{k.capitalize()}: {v:.2f}")

    if alert_message:
        print("\n--- Alert ---")
        print(alert_message)
        try:
            email = MIMEText(alert_message)
            email["Subject"] = "Crypto Risk Alert"
            email["From"] = EMAIL_USER
            email["To"] = ALERT_TO
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(email)
            print("\nAlert email sent successfully.")
        except Exception as e:
            print(f"\nFailed to send alert email: {e}")
    else:
        print("\nNo risk alerts triggered.")
