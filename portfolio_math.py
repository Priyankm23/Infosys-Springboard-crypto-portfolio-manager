import numpy as np
import pandas as pd
import sqlite3
from DB_portfolio import init_db, store_portfolio
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# for capping the weight to 50% if weights exceeds 50%
def cap_weights(weights, cap=0.5):
    capped = {s: min(w, cap) for s, w in weights.items()}
    total_capped = sum(capped.values())
    excess = 1 - total_capped
    if abs(excess) < 1e-9:
        return {s: round(capped[s], 6) for s in capped}
    uncapped_assets = [s for s in weights if weights[s] < cap]
    if not uncapped_assets:
        return {s: round(capped[s] / total_capped, 6) for s in capped}
    uncapped_total = sum(weights[s] for s in uncapped_assets)
    for s in uncapped_assets:
        add = (weights[s] / uncapped_total) * excess
        capped[s] += add
        if capped[s] > cap:
            capped[s] = cap
    total_final = sum(capped.values())
    return {s: round(capped[s] / total_final, 6) for s in capped}

# ---------- RULES ----------
def equal_weight(symbols):
    w = 1 / len(symbols)
    weights = {s: round(w, 6) for s in symbols}
    return cap_weights(weights)

def price_weight(symbols, prices):
    prices = {s: float(prices[s]) for s in symbols}
    total = sum(prices.values())
    weights = {s: prices[s] / total for s in symbols}
    return cap_weights(weights)

def inverse_volatility(symbols, returns):
    vols = {s: float(np.std(returns[s])) for s in symbols}
    inv = {s: 1 / vols[s] if vols[s] > 0 else 0 for s in symbols}
    total = sum(inv.values())
    weights = {s: inv[s] / total for s in symbols}
    return cap_weights(weights)

# ---------- HELPERS ----------
def percent_change(prices):
    changes = []
    for i in range(1, len(prices)):
        change = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
        changes.append(round(change, 6))
    return changes

def portfolio_return(weights, returns):
    min_len = min(len(r) for r in returns.values())
    port = []
    for i in range(min_len):
        r = sum(weights[s] * returns[s][i] for s in weights)
        port.append(round(r, 2))
    return port

def portfolio_risk(port):
    return round(np.std(port), 2)

# ---------- FETCH DATA ----------
def fetch_prices(uploaded_data=None):
    if uploaded_data is not None:
        data = {}
        for symbol, df in uploaded_data.items():
            df.columns = [col.strip().title() for col in df.columns]
            if 'Close' in df.columns and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date', 'Close'])
                df.set_index("Date", inplace=True)
                data[symbol.upper()] = df["Close"]
        prices_df = pd.concat(data, axis=1).dropna(how="any")
        return prices_df
    else:
        conn = sqlite3.connect("db/crypto.db")
        assets = ["btc_prices", "eth_prices", "usdc_prices"]
        data = {}
        for asset in assets:
            df = pd.read_sql(f"SELECT Close FROM {asset} ORDER BY Unix", conn)
            data[asset.replace("_prices", "").upper()] = df["Close"]
        conn.close()
        return pd.DataFrame(data)

# ---------- MAIN FUNCTION ----------
def run_and_plot_strategy(selected_rule="Equal", uploaded_data=None):
    prices_df = fetch_prices(uploaded_data)
    if prices_df.empty:
        raise ValueError("No price data available")

    returns = {col: percent_change(prices_df[col].tolist()) for col in prices_df.columns}
    prices = {col: prices_df[col].iloc[0] for col in prices_df.columns}
    symbols = list(returns.keys())

    rules = {
        "Equal": equal_weight,
        "Price": price_weight,
        "InvVol": inverse_volatility,
    }

    if selected_rule not in rules:
        raise ValueError(f"Unknown rule: {selected_rule}")

    if selected_rule == "Equal":
        w = rules[selected_rule](symbols)
    elif selected_rule == "Price":
        w = rules[selected_rule](symbols, prices)
    elif selected_rule == "InvVol":
        w = rules[selected_rule](symbols, returns)
    else:
        raise ValueError(f"Unknown rule: {selected_rule}")
    port_ret = portfolio_return(w, returns)

    n = min(15, len(port_ret))
    comparison_df = pd.DataFrame({f"{s}_Return": returns[s][:n] for s in symbols})
    comparison_df[f"{selected_rule}_Portfolio"] = port_ret[:n]

    # --- Plot returns ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for col in comparison_df.columns:
        ax1.plot(comparison_df[col], label=col, linewidth=2)
    ax1.set_title(f"{selected_rule} Portfolio vs Assets (15 Days)", fontsize=14)
    ax1.set_xlabel("Day Index")
    ax1.set_ylabel("% Return")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Plot cumulative returns ---
    comparison_cum = (1 + comparison_df / 100).cumprod() - 1
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for col in comparison_cum.columns:
        ax2.plot(comparison_cum[col], label=col, linewidth=2)
    ax2.set_title(f"{selected_rule} Portfolio vs Assets (Cumulative Returns)", fontsize=14)
    ax2.set_xlabel("Day Index")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # --- Insights ---
    insights = ""
    for col in comparison_df.columns:
        avg_ret = np.mean(comparison_df[col])
        risk = np.std(comparison_df[col])
        insights += f"{col} → Avg Return={avg_ret:.2f}%, Risk={risk:.2f}\n"

    store_portfolio(f"{selected_rule} Portfolio", port_ret, portfolio_risk(port_ret), w)

    return fig1, fig2, insights, w
