import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import warnings

# Suppress runtime warnings from NumPy when calculating log returns for empty shifts
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- DB Connection (SQLite) ----
DB_PATH = "db/crypto.db"

# ---- NEW Step 1: Combine uploaded DataFrames (Replaces get_prices) ----
def combine_uploaded_data(uploaded_data):
    """
    Combines a dictionary of asset DataFrames into a single price DataFrame.
    Assumes each DataFrame has a 'Date' column and a 'Close' or price-like column.
    Handles column standardization and data merging/cleaning.
    """
    if not uploaded_data:
        raise ValueError("Uploaded data is empty. Cannot run strategy.")

    all_dfs = []
    
    def identify_price_column(df):
        if 'Close' in df.columns:
            return 'Close'
        # Heuristic: assume the price column is the last non-Date, non-Volume column
        other_cols = [c for c in df.columns if c.lower() not in ['date', 'volume', 'adj close']]
        if other_cols:
            return other_cols[-1]
        raise ValueError("Uploaded DataFrame must contain a 'Date' or date-like column and a 'Close' or price-like column.")


    for coin, df in uploaded_data.items():
        df = df.copy()
        
        # Standardize column names (e.g., ensure 'Date' is capitalized if it exists)
        df.columns = [col.capitalize() for col in df.columns]
        
        if 'Date' not in df.columns:
            # Try to find a 'date' or 'timestamp' column
            date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
            if date_col:
                df = df.rename(columns={date_col: "Date"})
            else:
                raise ValueError(f"DataFrame for {coin} must contain a 'Date' or date-like column.")

        price_col = identify_price_column(df)

        df = df.rename(columns={price_col: coin, "Date": "date"})
        df = df[["date", coin]]
        # 'errors=coerce' handles problematic date strings, converting them to NaT
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df.dropna(subset=['date'])
        all_dfs.append(df)

    # Merge all coin price data on date
    prices_df = all_dfs[0]
    for df in all_dfs[1:]:
        prices_df = prices_df.merge(df, on="date", how="outer")

    prices_df = prices_df.sort_values("date").set_index("date").dropna(how='all')
    
    # Handle missing prices: forward fill then backward fill for merged data
    prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
    
    return prices_df

# ---- Step 3: Sharpe + Direct Sharpe weighting (UNCHANGED) ----
def sharpe_weights(prices_df, risk_free_rate=0.0):
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    
    if returns.empty:
        # Fallback for insufficient data
        raise ValueError("Insufficient data to calculate returns. Check for missing or single-row data.")
        
    mean_returns = returns.mean()
    vol = returns.std()

    # Calculation of Daily Sharpe Ratio
    sharpe_ratios = (mean_returns - risk_free_rate) / vol
    sharpe_ratios = sharpe_ratios.clip(lower=0)  # avoid negatives

    print("\n[SHARPE RATIOS (Daily)]")
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

# ---- Step 4: Dynamic weights + portfolio return (REFRACTORED) ----
def dynamic_weights_and_return(uploaded_data):
    """
    Computes dynamic weights and the latest portfolio return from uploaded data.
    """
    prices = combine_uploaded_data(uploaded_data)

    # Compute weights
    weights, returns = sharpe_weights(prices)

    # Portfolio return (latest day) - uses the last computed daily return
    # The 'latest' return is the last one in the returns DataFrame, based on the historical data.
    if returns.empty:
        portfolio_return = 0.0
        print("\n[PORTFOLIO RETURN] Latest Day: 0.00% (Insufficient data for latest return)")
    else:
        portfolio_return = np.dot(weights, returns.iloc[-1])
        print(f"\n[PORTFOLIO RETURN] Latest Day: {portfolio_return:.4%}")

    return dict(weights), portfolio_return, returns

# ---- Step 5: Stress Test (REFRACTORED for dynamic asset list) ----
def stress_test(weights, n=1000):
    """
    Simulates portfolio returns across Bull, Bear, and Volatile scenarios 
    for all assets in the portfolio dynamically.
    """
    weights = {k.lower(): v for k, v in weights.items()}
    assets = list(weights.keys())

    # Simplified generic distributions for all assets in the portfolio
    scenarios_params = {
        # High mean, low volatility (e.g., 4% average daily return)
        "Bull Market": (0.04, 0.01),      
        # Negative mean, medium volatility (e.g., -4% average daily return)
        "Bear Market": (-0.04, 0.015),    
        # Zero mean, high volatility (e.g., 0% average return but 8% daily vol)
        "Volatile Market": (0.00, 0.08)   
    }

    simulated_scenarios = {}
    for scenario, (mu, sigma) in scenarios_params.items():
        scenario_df_data = {}
        for asset in assets:
            # All assets follow the same market distribution for the test
            scenario_df_data[asset] = np.random.normal(mu, sigma, n)
        simulated_scenarios[scenario] = pd.DataFrame(scenario_df_data)

    results = {}
    for scenario, df in simulated_scenarios.items():
        weight_vector = np.array([weights.get(col, 0) for col in df.columns])
        portfolio_returns = df.dot(weight_vector)

        results[scenario] = {
            "mean_return": float(portfolio_returns.mean()),
            "volatility": float(portfolio_returns.std()),
            "min_return": float(portfolio_returns.min()),
            "max_return": float(portfolio_returns.max())
        }

    return results

# ---- Step 6: Interpret Stress Test (Slightly adjusted for generic portfolio) ----
def interpret_stress_test(results):
    insights = []

    bear = results["Bear Market"]
    insights.append(f"Bear Market → Avg: {bear['mean_return']:.2%}, Worst-case: {bear['min_return']:.2%}. "
                    f"Portfolio faces a downside, and diversification will determine the severity of the loss.")

    bull = results["Bull Market"]
    insights.append(f"Bull Market → Avg: {bull['mean_return']:.2%}, Best-case: {bull['max_return']:.2%}. "
                    f"The current weights allow the portfolio to capture significant upside potential.")

    vol = results["Volatile Market"]
    insights.append(f"Volatile Market → Range: {vol['min_return']:.2%} to {vol['max_return']:.2%}. "  
                    f"High swings show significant uncertainty, with potential for both large gains and losses.")

    return insights

# ---- Step 7: Store weights + portfolio return in DB (UNCHANGED) ----
def store_weights(weights_dict, portfolio_return):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    valid_weights = {k: v for k, v in weights_dict.items()}

    columns_sql = ", ".join([f"{coin} REAL" for coin in valid_weights.keys()])
    
    if not columns_sql:
        print("\n[DB] No weights to store.")
        conn.close()
        return

    try:
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS weights_history (
                date TEXT PRIMARY KEY,
                portfolio_return REAL,
                {columns_sql}
            )
        """)
    except sqlite3.OperationalError as e:
        print(f"\n[DB WARNING] Could not create/update weights_history table due to schema mismatch: {e}")
        conn.close()
        return

    columns = ", ".join(["date", "portfolio_return"] + list(valid_weights.keys()))
    placeholders = ", ".join("?" for _ in range(len(valid_weights) + 2))
    values = [datetime.now().strftime("%Y-%m-%d"), portfolio_return] + list(valid_weights.values())

    if len(columns.split(',')) != len(values):
        print("\n[DB WARNING] Column/Value count mismatch. Skipping DB insert.")
        conn.close()
        return

    try:
        c.execute(f"INSERT OR REPLACE INTO weights_history ({columns}) VALUES ({placeholders})", values)
        conn.commit()
        print("\n[DB] Weights & Portfolio return stored successfully.")
    except sqlite3.OperationalError as e:
         print(f"\n[DB WARNING] Could not insert into weights_history: {e}")
    finally:
        conn.close()

# ---- Step 8: Run the complete process (REFRACTORED to require uploaded_data) ----
def run_investment_strategy(uploaded_data):
    """ 
    Runs the complete investment strategy process:
    Computes dynamic weights from uploaded data, Performs stress testing, 
    Interprets results and Stores the new weights.
    Returns all results for display.
    """
    weights, port_return, returns = dynamic_weights_and_return(uploaded_data)

    results = stress_test(weights)
    insights = interpret_stress_test(results)

    # Store weights (even if derived from uploaded, it's a good practice)
    store_weights(weights, port_return)
    
    return {
        "weights": weights,
        "portfolio_return": port_return,
        "stress_test_results": results,
        "insights": insights
    }