import numpy as np
import sqlite3
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

#percentage change function
def percent_change(prices):
    changes=[]
    for i in range(1,len(prices)):
        change= ((prices[i] - prices[i-1])/prices[i-1])*100
        changes.append(round(change,2))
    return changes


def moving_change(prices,window_size):
    averages = [None] * (window_size - 1)
    for i in range(len(prices) - window_size + 1):
        window = prices[i:i + window_size]
        avg = sum(window) / window_size
        averages.append(round(avg,2))
    
    return averages


def calculate_volatility(prices):
    changes = percent_change(prices)
    volatility = np.std(changes)  
    return round(volatility, 2)


def rolling_volatility(prices, window_size):
    volatilities = [None] * (window_size - 1)
    for i in range(len(prices) - window_size + 1):
        window = prices[i:i + window_size]
        pchanges = percent_change(window)
        vol = np.std(pchanges)
        volatilities.append(round(vol, 2))
    return volatilities

def trading_signals(prices):
    changes = percent_change(prices)
    signals=[]
    for change in changes:
        if change>2:               
            signals.append("BUY")
        elif change<-2:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    return signals

def portfolio_return(assetA, assetB, wA=0.5, wB=0.2):
    returns = []
    for a, b in zip(assetA, assetB):
        r = wA * a + wB * b
        returns.append(round(r, 2))
    return returns


def task(symbol, df):
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    prices = df['Close'].tolist()
    if 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'], dayfirst=True)
        dates = df['date'].tolist()
    elif 'Unix' in df.columns:
        dates = df['Unix'].tolist()

    # Calculate metrics
    pchanges = percent_change(prices)
    mchanges = moving_change(prices, 3)
    volatilities = rolling_volatility(prices, 7) 
    signals = trading_signals(prices)

    # Align with dates 
    daily_data = []
    for i in range(1, len(prices)):
        entry = {
            "symbol": symbol,
            "date": dates[i],               
            "percent_change": pchanges[i-1],
            "signal": signals[i-1],
            "moving_avg": mchanges[i-1],
            "volatility": volatilities[i-1]
        }
        daily_data.append(entry)

    daily_df = pd.DataFrame(daily_data)
    return daily_df


def run_metric_calculation(uploaded_data=None, coins=[("BTC", "btc_prices"), ("ETH", "eth_prices")]):
    """
    Calculates and stores technical metrics for a given list of coins.
    Returns a dictionary with a status message and the calculated dataframes.
    """
    if uploaded_data:
        with ThreadPoolExecutor(max_workers=len(uploaded_data)) as executor:
            results = list(executor.map(lambda item: task(item[0], item[1]), uploaded_data.items()))
        
        dataframes = {}
        for df in results:
            if not df.empty:
                symbol = df['symbol'].iloc[0]
                dataframes[symbol] = df
        return {"message": "Metrics calculated for uploaded data.", "dataframes": dataframes}
    else:
        def db_task(symbol, table):
            conn = sqlite3.connect("db/crypto.db", check_same_thread=False)
            query = f"SELECT Unix, Close FROM {table} ORDER BY Unix;"  
            df = pd.read_sql(query, conn)
            conn.close()
            return task(symbol, df)

        with ThreadPoolExecutor(max_workers=len(coins)) as executor:
            results = list(executor.map(lambda args: db_task(*args), coins))
            
        conn = sqlite3.connect("db/crypto.db")
        messages = []
        dataframes = {}
        for df in results:
            if not df.empty:
                symbol = df['symbol'].iloc[0]
                df.to_sql("crypto_metrics", conn, if_exists="append", index=False)
                message = f"Inserted {len(df)} rows of metrics for {symbol} into the database."
                print(message)
                messages.append(message)
                dataframes[symbol] = df
        conn.close()
        
        final_message = "\n".join(messages)
        print("\nAll daily metrics stored in crypto_metrics table using parallel tasking!")
        return {"message": final_message, "dataframes": dataframes}

if __name__ == "__main__":
    output = run_metric_calculation()
    print(output["message"])