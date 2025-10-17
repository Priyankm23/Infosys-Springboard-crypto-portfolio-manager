import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV # Added LassoCV as an alternative option
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# Feature Engineering (IMPROVED: Added a .shift(1) to all MA features)
# -----------------------
def create_features(df):
    df['Prev_Close'] = df['Close'].shift(1)
    # Target variable: Return for the current day (Close price change from yesterday's close)
    df['Return'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
    
    # IMPROVEMENT: Lag MA features by 1 day to ensure they only contain
    # information available BEFORE the current day's Return is known.
    # This mitigates look-ahead bias and reduces artificial inflation of Train R^2.
    df['MA3'] = df['Close'].rolling(3).mean().shift(1)
    df['MA7'] = df['Close'].rolling(7).mean().shift(1)
    df['MA14'] = df['Close'].rolling(14).mean().shift(1)
    
    df.dropna(inplace=True)
    return df

# -----------------------
# Rolling Ridge Regression (IMPROVED: Increased alpha range for stronger regularization)
# -----------------------
def rolling_ridge(df, window=90):
    features = ['Open', 'High', 'Low', 'Prev_Close', 'MA3', 'MA7', 'MA14']
    X = df[features]
    y = df['Return']
    scaler = StandardScaler()

    preds = []
    actuals = []
    train_r2_list = []

    # IMPROVEMENT: Increased the upper bound of alpha from 10^3 to 10^5 
    # to allow for stronger regularization (smaller coefficients) and reduce overfitting.
    alphas = np.logspace(-3, 5, 15) 

    for i in range(window, len(df)):
        X_train = X.iloc[i-window:i]
        y_train = y.iloc[i-window:i]
        X_test = X.iloc[i:i+1]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use RidgeCV for finding the best alpha
        model = RidgeCV(alphas=alphas, store_cv_results=True)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        preds.append(y_pred[0])
        actuals.append(y.iloc[i])

        y_train_pred = model.predict(X_train_scaled)
        train_r2_list.append(r2_score(y_train, y_train_pred))

    df_out = df.iloc[window:].copy()
    df_out['Predicted_Return'] = preds
    df_out['Actual_Return'] = actuals
    df_out['Train_R2'] = train_r2_list
    return df_out

# -----------------------
# Evaluate metrics (No Change)
# -----------------------
def evaluate_results(df):
    test_r2 = r2_score(df['Actual_Return'], df['Predicted_Return'])
    test_mse = mean_squared_error(df['Actual_Return'], df['Predicted_Return'])
    last_actual = df['Actual_Return'].iloc[-1] * 100
    last_pred = df['Predicted_Return'].iloc[-1] * 100
    avg_train_r2 = np.mean(df['Train_R2'])
    return {
        'Train R² Score': round(avg_train_r2, 4),
        'Test R² Score': round(test_r2, 4),
        'Test MSE': round(test_mse, 6),
        'Last Actual Return (%)': round(last_actual, 4),
        'Last Predicted Return (%)': round(last_pred, 4)
    }

# -----------------------
# Plotting (No Change)
# -----------------------
def plot_predictions(df, label):
    plt.figure(figsize=(10,5))
    plt.plot(df['Actual_Return'].values, label="Actual Return", linewidth=2)
    plt.plot(df['Predicted_Return'].values, label="Predicted Return", linestyle="--", linewidth=2)
    plt.title(f"Actual vs Predicted Returns ({label})")
    plt.xlabel("Time (Days)")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig = plt.gcf()
    plt.close(fig)
    return fig

# -----------------------
# Main for Streamlit (No Change)
# -----------------------
def main(dataframes):
    results = []
    figures = []
    processed_data = {}

    window = 90
    
    # Step 1: Process individual assets
    for label, df in dataframes.items():
        try:
            df = df[['Open', 'High', 'Low', 'Close']].copy()
            df = create_features(df)  # ensures 'Return' exists

            if len(df) <= window:
                print(f"⚠️ Not enough data points for {label}")
                continue

            df_out = rolling_ridge(df, window=window)  # predict returns
            processed_data[label] = df_out

            # Evaluate individual asset
            metrics = evaluate_results(df_out)
            fig = plot_predictions(df_out, label)

            results.append({
                'label': label,
                'train_r2': metrics['Train R² Score'],
                'test_r2': metrics['Test R² Score'],
                'test_mse': metrics['Test MSE'],
                'last_actual': metrics['Last Actual Return (%)']/100,
                'last_predicted': metrics['Last Predicted Return (%)']/100
            })
            figures.append(fig)

        except Exception as e:
            print(f"❌ Error during prediction for {label}: {e}")

    # Step 2: Portfolio (Sharpe-based allocation)
    if processed_data:
        min_len = min(len(df) for df in processed_data.values())
        features = ['Open','High','Low','Prev_Close','MA3','MA7','MA14']
        portfolio_data = {f: np.zeros(min_len) for f in features}
        portfolio_return = np.zeros(min_len)
    
        # Compute weights based on volatility
        vol_dict = {k: df['Actual_Return'].std() for k, df in processed_data.items()}
        # Use a small constant for return std to prevent division by zero if all returns are the same
        vol_dict = {k: v if v > 1e-6 else 1e-6 for k, v in vol_dict.items()} 
        total_inv_vol = sum(1/v for v in vol_dict.values())
        weights = {k: (1/v)/total_inv_vol for k,v in vol_dict.items()}
    
        # Combine features and returns
        for label, df in processed_data.items():
            df_trim = df.iloc[-min_len:]
            w = weights[label]
            for f in features:
                # Use current day features from the asset, weighted
                portfolio_data[f] += df_trim.index.get_level_values(0).map(lambda x: df.loc[x, f] * w).values 
            
            # Weighted average of actual returns
            portfolio_return += df_trim['Actual_Return'].values * w
    
        portfolio_df = pd.DataFrame(portfolio_data, index=df_trim.index)
        portfolio_df['Return'] = portfolio_return
    
        # Portfolio prediction
        portfolio_df_out = rolling_ridge(portfolio_df, window=window)
        metrics = evaluate_results(portfolio_df_out)
        fig = plot_predictions(portfolio_df_out, "Portfolio (Sharpe Allocated)")

        results.insert(0,{
            'label': "Portfolio (Sharpe Allocated)",
            'train_r2': metrics['Train R² Score'],
            'test_r2': metrics['Test R² Score'],
            'test_mse': metrics['Test MSE'],
            'last_actual': metrics['Last Actual Return (%)']/100,
            'last_predicted': metrics['Last Predicted Return (%)']/100
        })
        figures.insert(0, fig)

    return {'results': results, 'figures': figures}