import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime

# ---- DB Connection (SQLite) ----
DB_PATH = "db/crypto.db"

# ---- Step 1: Load historical data from DB ----
def get_prices(coin_ids):
    conn = sqlite3.connect(DB_PATH)
    all_dfs = []

    for coin in coin_ids:
        table_name = f"{coin}_prices"
        query = f"SELECT Date, Close FROM {table_name} ORDER BY Date ASC"
        df = pd.read_sql(query, conn)
        df = df.rename(columns={"Close": coin, "Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        all_dfs.append(df)

    conn.close()

    # Merge all coin price data on date
    prices_df = all_dfs[0]
    for df in all_dfs[1:]:
        prices_df = prices_df.merge(df, on="date", how="outer")

    prices_df = prices_df.sort_values("date").set_index("date")
    return prices_df

# ---- Step 2: Fetch live price from CoinGecko ----
def fetch_live_price(coin_id, vs_currency="usd"):
    coin_map = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "usdc": "usd-coin"
    }
    cg_id = coin_map.get(coin_id.lower())
    if not cg_id:
        raise ValueError(f"No CoinGecko mapping for coin: {coin_id}")

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": cg_id, "vs_currencies": vs_currency}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    live_price = r.json()[cg_id][vs_currency]
    print(f"[LIVE PRICE] {coin_id.upper()}: {live_price} {vs_currency.upper()}")
    return live_price

# ---- Step 3: Sharpe + Direct Sharpe weighting ----
def sharpe_weights(prices_df, risk_free_rate=0.0):
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    mean_returns = returns.mean()
    vol = returns.std()

    sharpe_ratios = (mean_returns - risk_free_rate) / vol
    sharpe_ratios = sharpe_ratios.clip(lower=0)  # avoid negatives

    print("\n[SHARPE RATIOS]")
    for asset, val in sharpe_ratios.items():
        print(f"{asset}: {val:.4f}")

    # Normalize Sharpe ratios into weights (sum = 1)
    if sharpe_ratios.sum() > 0:
        weights = sharpe_ratios / sharpe_ratios.sum()
    else:
        # fallback: equal allocation
        weights = pd.Series([1/len(sharpe_ratios)]*len(sharpe_ratios), index=sharpe_ratios.index)

    print("\n[WEIGHTS (Sharpe ratio normalized)]")
    for asset, val in weights.items():
        print(f"{asset}: {val:.4f}")

    return weights, returns

# ---- Step 4: Dynamic weights + portfolio return ----
def dynamic_weights_and_return(coin_ids, vs_currency="usd"):
    prices = get_prices(coin_ids)

    # Fetch live prices and append to latest date
    latest_prices = {}
    for cid in coin_ids:
        latest_prices[cid] = fetch_live_price(cid, vs_currency)

    today = pd.DataFrame([latest_prices], index=[pd.Timestamp.today().normalize()])
    prices = pd.concat([prices, today], axis=0)

    # Compute weights
    weights, returns = sharpe_weights(prices)

    # Portfolio return (latest day)
    portfolio_return = np.dot(weights, returns.iloc[-1])
    print(f"\n[PORTFOLIO RETURN] Latest Day: {portfolio_return:.4%}")

    return dict(weights), portfolio_return, returns

# ---- Step 5: Stress Test ----
def stress_test(weights, n=1000):
    # Normalize weights
    weights = {k.lower(): v for k, v in weights.items()}

    scenarios = {
        "Bull Market": pd.DataFrame({
            "btc": np.random.normal(0.04, 0.01, n),
            "eth": np.random.normal(0.03, 0.015, n),
            "usdc": np.random.normal(0.001, 0.001, n)
        }),
        "Bear Market": pd.DataFrame({
            "btc": np.random.normal(-0.04, 0.015, n),
            "eth": np.random.normal(-0.035, 0.02, n),
            "usdc": np.random.normal(0.001, 0.001, n)
        }),
        "Volatile Market": pd.DataFrame({
            "btc": np.random.normal(0.0, 0.08, n),
            "eth": np.random.normal(0.0, 0.09, n),
            "usdc": np.random.normal(0.001, 0.001, n)
        })
    }

    results = {}
    for scenario, df in scenarios.items():
        weight_vector = np.array([weights.get(col, 0) for col in df.columns])
        portfolio_returns = df.dot(weight_vector)

        results[scenario] = {
            "mean_return": float(portfolio_returns.mean()),
            "volatility": float(portfolio_returns.std()),
            "min_return": float(portfolio_returns.min()),
            "max_return": float(portfolio_returns.max())
        }

    return results

# ---- Step 6: Interpret Stress Test ----
def interpret_stress_test(results):
    insights = []

    bear = results["Bear Market"]
    if bear["mean_return"] < 0:
        insights.append(f"Bear Market → Avg: {bear['mean_return']:.2%}, Worst-case: {bear['min_return']:.2%}. "
                        f"Portfolio faces downside but diversification cushions extreme losses.")

    bull = results["Bull Market"]
    insights.append(f"Bull Market → Avg: {bull['mean_return']:.2%}, Best-case: {bull['max_return']:.2%}. "
                    f"Portfolio captures upside potential effectively.")

    vol = results["Volatile Market"]
    insights.append(f"Volatile Market → Range: {vol['min_return']:.2%} to {vol['max_return']:.2%}. "    
                    f"High swings show both strong growth opportunities and high risk.")

    return insights

# ---- Step 7: Store weights + portfolio return in DB ----
def store_weights(weights_dict, portfolio_return):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create table if not exists
    columns_sql = ", ".join([f"{coin} REAL" for coin in weights_dict.keys()])
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS weights_history (
            date TEXT PRIMARY KEY,
            portfolio_return REAL,
            {columns_sql}
        )
    """)

    columns = ", ".join(["date", "portfolio_return"] + list(weights_dict.keys()))
    placeholders = ", ".join("?" for _ in range(len(weights_dict) + 2))
    values = [datetime.now().strftime("%Y-%m-%d"), portfolio_return] + list(weights_dict.values())

    c.execute(f"INSERT OR REPLACE INTO weights_history ({columns}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()
    print("\n[DB] Weights & Portfolio return stored successfully.")

def run_investment_strategy(coin_ids=["btc", "eth", "usdc"]):
    """ 
    Runs the complete investment strategy process:
    Fetches dynamic weights, Performs stress testing, Interprets results and Stores the new weights.
    Returns all results for display.
    """
    weights, port_return, returns = dynamic_weights_and_return(coin_ids)

    results = stress_test(weights)
    insights = interpret_stress_test(results)

    store_weights(weights, port_return)
    
    return {
        "weights": weights,
        "portfolio_return": port_return,
        "stress_test_results": results,
        "insights": insights
    }

# ---- Step 8: Run the process ----
if __name__ == "__main__":
    output = run_investment_strategy()

    print("\n[WEIGHTS]")
    print(output["weights"])
    
    print("\n[PORTFOLIO RETURN]")
    print(output["portfolio_return"])

    print("\n[STRESS TEST RESULTS]")
    for scenario, stats in output["stress_test_results"].items():
        print(scenario, stats)

    print("\n[INSIGHTS]")
    for line in output["insights"]:
        print(line)
