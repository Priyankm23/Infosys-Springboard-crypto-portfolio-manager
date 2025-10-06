import numpy as np
import pandas as pd
import sqlite3
from DB_portfolio import init_db, store_portfolio
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# for capping the weight to 50% if weights exceeds 50%
def cap_weights(weights, cap=0.5):
    # Step 1: Cap all
    capped = {s: min(w, cap) for s, w in weights.items()}

    # Step 2: Compute excess
    total_capped = sum(capped.values())
    excess = 1 - total_capped

    if abs(excess) < 1e-9:
        return {s: round(capped[s], 6) for s in capped}

    # Step 3: Find assets below cap
    uncapped_assets = [s for s in weights if weights[s] < cap]

    if not uncapped_assets:
        # if all capped, just normalize
        return {s: round(capped[s] / total_capped, 6) for s in capped}

    # Step 4: Distribute excess proportionally based on original weights
    uncapped_total = sum(weights[s] for s in uncapped_assets)
    for s in uncapped_assets:
        add = (weights[s] / uncapped_total) * excess
        capped[s] += add
        if capped[s] > cap:  # recheck cap
            capped[s] = cap

    # Step 5: Final normalization
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
    inv = {s: 1/vols[s] if vols[s] > 0 else 0 for s in symbols}
    total = sum(inv.values())
    weights = {s: inv[s] / total for s in symbols}
    return cap_weights(weights)

# ---------- HELPERS ----------
def percent_change(prices):
    changes=[]
    for i in range(1,len(prices)):
        change= ((prices[i] - prices[i-1])/prices[i-1])*100
        changes.append(round(change,6))
    return changes

def portfolio_return(weights, returns):
    min_len = min(len(r) for r in returns.values())
    port = []
    for i in range(min_len):
        r = sum(weights[s]*returns[s][i] for s in weights)
        port.append(round(r,2))
    return port

def portfolio_risk(port):
    return round(np.std(port),2)

# ---------- FUNCTION TO PROCESS ONE RULE ----------
def process_rule(rule_name, weights_func, symbols, prices, returns):
    if rule_name == "Equal":
        w = weights_func(symbols)
    elif rule_name == "Price":
        w = weights_func(symbols, prices)
    elif rule_name == "InvVol":
        w = weights_func(symbols, returns)
    else:
        return None

    port = portfolio_return(w, returns)
    risk = portfolio_risk(port)

    store_portfolio(f"{rule_name} Portfolio", port, risk, w)
    return f"{rule_name} completed"

def run_and_plot_strategy(selected_rule="Equal"):
    """
    Runs the analysis for a selected portfolio strategy, generates plots,
    and returns the figures and insight text.
    """
    conn = sqlite3.connect("db/crypto.db")
    btc = pd.read_sql("SELECT Close FROM btc_prices ORDER BY Unix", conn)
    eth = pd.read_sql("SELECT Close FROM eth_prices ORDER BY Unix", conn)
    usdc = pd.read_sql("SELECT Close FROM usdc_prices ORDER BY Unix", conn)
    conn.close()

    returns = {
        "BTC": percent_change(btc['Close'].tolist()),
        "ETH": percent_change(eth['Close'].tolist()),
        "USDC": percent_change(usdc['Close'].tolist())
    }
    prices = {"BTC": btc['Close'].iloc[0], "ETH": eth['Close'].iloc[0], "USDC": usdc['Close'].iloc[0]}
    symbols = list(returns.keys())

    rules = {
        "Equal": equal_weight,
        "Price": price_weight,
        "InvVol": inverse_volatility,
    }

    # --- Run selected rule ---
    if selected_rule == "Equal":
        w = rules[selected_rule](symbols)
    elif selected_rule == "Price":
        w = rules[selected_rule](symbols, prices)
    elif selected_rule == "InvVol":
        w = rules[selected_rule](symbols, returns)
    else:
        raise ValueError(f"Unknown rule: {selected_rule}")
    port_ret = portfolio_return(w, returns)
    
    # --- Generate Comparison Data (15 days) ---
    n = 15
    comparison_df = pd.DataFrame({
        "BTC_Return": returns["BTC"][:n],
        "ETH_Return": returns["ETH"][:n],
        "USDC_Return": returns["USDC"][:n],
        f"{selected_rule}_Portfolio": port_ret[:n],
    })
    
    # --- Generate Plots ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(comparison_df["BTC_Return"], label="BTC", color="orange", linewidth=2)
    ax1.plot(comparison_df["ETH_Return"], label="ETH", color="blue", linewidth=2)
    ax1.plot(comparison_df["USDC_Return"], label="USDC", color="green", linewidth=2)
    ax1.plot(comparison_df[f"{selected_rule}_Portfolio"], label=f"{selected_rule} Portfolio", color="red", linewidth=2)
    ax1.set_title(f"{selected_rule} Portfolio vs Assets (15 Days)", fontsize=14)
    ax1.set_xlabel("Day Index")
    ax1.set_ylabel("% Return")
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.tight_layout()

    comparison_cum = (1 + comparison_df / 100).cumprod() - 1
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(comparison_cum["BTC_Return"], label="BTC", color="orange", linewidth=2)
    ax2.plot(comparison_cum["ETH_Return"], label="ETH", color="blue", linewidth=2)
    ax2.plot(comparison_cum["USDC_Return"], label="USDC", color="green", linewidth=2)
    ax2.plot(comparison_cum[f"{selected_rule}_Portfolio"], label=f"{selected_rule} Portfolio", color="red", linewidth=2)
    ax2.set_title(f"{selected_rule} Portfolio vs Assets (Cumulative Returns)", fontsize=14)
    ax2.set_xlabel("Day Index")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()

    # --- Generate Insights ---
    insights = ""
    metrics = {}
    for col in ["BTC_Return", "ETH_Return", "USDC_Return", f"{selected_rule}_Portfolio"]:
        data = comparison_df[col].dropna().tolist()
        avg_ret = np.mean(data)
        risk = np.std(data)
        metrics[col] = (avg_ret, risk)
        if col.endswith("_Portfolio"):
            insights += f"{col} → Avg Return={avg_ret:.2f}%, Risk={risk:.2f} → Balanced allocation (capped ≤50%).\n"
        else:
            insights += f"{col.replace('_Return','')} → Avg Return={avg_ret:.2f}%, Risk={risk:.2f}\n"

    return fig1, fig2, insights

# ---------- MAIN ----------
if __name__ == "__main__":
    init_db()
    
    # Example of running all rules and storing them
    conn = sqlite3.connect("db/crypto.db")
    btc = pd.read_sql("SELECT Close FROM btc_prices ORDER BY Unix", conn)
    eth = pd.read_sql("SELECT Close FROM eth_prices ORDER BY Unix", conn)
    usdc = pd.read_sql("SELECT Close FROM usdc_prices ORDER BY Unix", conn)
    conn.close()

    returns = {
        "BTC": percent_change(btc['Close'].tolist()),
        "ETH": percent_change(eth['Close'].tolist()),
        "USDC": percent_change(usdc['Close'].tolist())
    }
    prices = {"BTC": btc['Close'].iloc[0], "ETH": eth['Close'].iloc[0], "USDC": usdc['Close'].iloc[0]}
    symbols = list(returns.keys())

    rules = {
        "Equal": equal_weight,
        "Price": price_weight,
        "InvVol": inverse_volatility,
    }

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_rule, name, func, symbols, prices, returns)
            for name, func in rules.items()
        ]
        for future in as_completed(futures):
            print(future.result())

    # Example of running the plotting function
    fig1, fig2, insights = run_and_plot_strategy("Equal")
    print("\n--- Insights from 15 Days ---")
    print(insights)
    plt.show() # Show plots when run as a script

