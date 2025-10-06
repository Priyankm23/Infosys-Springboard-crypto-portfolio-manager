import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Load data from DB ===
def load_data(table_name):
    conn = sqlite3.connect("db/crypto.db")
    query = f"SELECT Date, Open, Close, High, Low FROM {table_name} ORDER BY Date ASC"
    df = pd.read_sql_query(query, conn, parse_dates=["Date"])
    conn.close()
    df.set_index("Date", inplace=True)
    return df

# === Feature engineering for an asset ===
def create_features(df):
    df['Prev_Close'] = df['Close'].shift(1)
    df['Return'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
    df['MA3'] = df['Close'].rolling(3).mean()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df.dropna(inplace=True)
    return df

# === Remove outliers - Removes extreme values beyond ±3σ to prevent them from distorting the model.===
def remove_outliers(df, col='Return', threshold=3):
    mean, std = df[col].mean(), df[col].std()
    return df[(df[col] > mean - threshold * std) & (df[col] < mean + threshold * std)]

def train_predict(df, label, split_ratio=0.8, alpha=1.0):
    X = df[['Open', 'Prev_Close', 'High', 'Low', 'MA3', 'MA7', 'MA14']]
    y = df['Return']

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Time-based split
    split_index = int(len(X_scaled) * split_ratio)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    df['Predicted'] = np.nan
    df.iloc[split_index:, df.columns.get_loc('Predicted')] = y_test_pred

    results = {
        "label": label,
        "selected_features": list(selected_features),
        "train_r2": model.score(X_train, y_train),
        "test_r2": model.score(X_test, y_test),
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "last_actual": y.iloc[-1],
        "last_predicted": df['Predicted'].iloc[-1]
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_test_pred, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
    ax.set_xlabel("Actual Returns")
    ax.set_ylabel("Predicted Returns")
    ax.set_title(f"{label} Predicted vs Actual Returns")
    ax.grid(True)

    return df, y, model, selected_features, X, fig, results

# === Main ===
def main():
    weights = {"BTC": 0.5, "ETH": 0.3, "USDC": 0.2}
    all_results = []
    all_figs = []

    # BTC
    btc_df = create_features(load_data("btc_prices"))
    btc_df = remove_outliers(btc_df)
    btc_df, btc_y, btc_model, btc_features, btc_X, fig, results = train_predict(btc_df, "BTC", alpha=0.5)
    all_figs.append(fig)
    all_results.append(results)

    # ETH
    eth_df = create_features(load_data("eth_prices"))
    eth_df = remove_outliers(eth_df)
    eth_df, eth_y, eth_model, eth_features, eth_X, fig, results = train_predict(eth_df, "ETH", alpha=0.5)
    all_figs.append(fig)
    all_results.append(results)

    # USDC
    usdc_df = create_features(load_data("usdc_prices"))
    usdc_df = remove_outliers(usdc_df)
    usdc_df, usdc_y, usdc_model, usdc_features, usdc_X, fig, results = train_predict(usdc_df, "USDC", alpha=0.5)
    all_figs.append(fig)
    all_results.append(results)

    # === Portfolio ===
    portfolio_df = pd.DataFrame(index=btc_df.index)
    portfolio_df['Return'] = btc_y * weights["BTC"] + eth_y * weights["ETH"] + usdc_y * weights["USDC"]
    portfolio_df['Open'] = btc_X['Open'] * weights["BTC"] + eth_X['Open'] * weights["ETH"] + usdc_X['Open'] * weights["USDC"]
    portfolio_df['Prev_Close'] = btc_X['Prev_Close'] * weights["BTC"] + eth_X['Prev_Close'] * weights["ETH"] + usdc_X['Prev_Close'] * weights["USDC"]
    portfolio_df['High'] = btc_X['High'] * weights["BTC"] + eth_X['High'] * weights["ETH"] + usdc_X['High'] * weights["USDC"]
    portfolio_df['Low'] = btc_X['Low'] * weights["BTC"] + eth_X['Low'] * weights["ETH"] + usdc_X['Low'] * weights["USDC"]
    portfolio_df['MA3'] = btc_X['MA3'] * weights["BTC"] + eth_X['MA3'] * weights["ETH"] + usdc_X['MA3'] * weights["USDC"]
    portfolio_df['MA7'] = btc_X['MA7'] * weights["BTC"] + eth_X['MA7'] * weights["ETH"] + usdc_X['MA7'] * weights["USDC"]
    portfolio_df['MA14'] = btc_X['MA14'] * weights["BTC"] + eth_X['MA14'] * weights["ETH"] + usdc_X['MA14'] * weights["USDC"]

    portfolio_df.dropna(inplace=True)
    split_ratio = 0.8
    split_index = int(len(portfolio_df) * split_ratio)
    X_portfolio = portfolio_df[['Open', 'Prev_Close', 'High', 'Low', 'MA3', 'MA7', 'MA14']]
    y_portfolio = portfolio_df['Return']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_portfolio)

    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_portfolio.iloc[:split_index], y_portfolio.iloc[split_index:]

    model_portfolio = Ridge(alpha=0.5)
    model_portfolio.fit(X_train, y_train)

    y_train_pred = model_portfolio.predict(X_train)
    y_test_pred = model_portfolio.predict(X_test)

    portfolio_df['Predicted'] = np.nan
    portfolio_df.iloc[split_index:, portfolio_df.columns.get_loc('Predicted')] = y_test_pred

    port_results = {
        "label": "Portfolio",
        "train_r2": model_portfolio.score(X_train, y_train),
        "test_r2": model_portfolio.score(X_test, y_test),
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "last_actual": y_portfolio.iloc[-1],
        "last_predicted": portfolio_df['Predicted'].iloc[-1]
    }
    all_results.append(port_results)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_test_pred, alpha=0.6)
    ax.plot([y_portfolio.min(), y_portfolio.max()], [y_portfolio.min(), y_portfolio.max()], color='red', linewidth=2)
    ax.set_xlabel("Actual Returns")
    ax.set_ylabel("Predicted Returns")
    ax.set_title("Portfolio Predicted vs Actual Returns")
    ax.grid(True)
    all_figs.append(fig)

    return {"results": all_results, "figures": all_figs}

if __name__ == "__main__":
    output = main()
    for res in output['results']:
        print(f"\n=== {res['label']} ===")
        if res.get('selected_features'):
            print(f"Selected Features: {res['selected_features']}")
        print(f"Train R² Score: {res['train_r2']:.4f}")
        print(f"Test R² Score: {res['test_r2']:.4f}")
        print(f"Train MSE: {res['train_mse']:.6f}")
        print(f"Test MSE: {res['test_mse']:.6f}")
        print(f"Last Actual Return: {res['last_actual']:.4%}")
        print(f"Last Predicted Return: {res['last_predicted']:.4%}")
    plt.show() # Show all plots when run as a script
