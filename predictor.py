import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# RSI Computation
# -----------------------
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -----------------------
# Feature Engineering
# -----------------------
def create_features(df):
    df['Prev_Close'] = df['Close'].shift(1)
    df['Return'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
    df['MA3'] = df['Close'].rolling(3).mean()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df['Volatility'] = df['Return'].rolling(7).std()
    df['Momentum'] = df['Close'] / df['Close'].shift(3) - 1
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

# -----------------------
# Preprocessing & Feature Selection
# -----------------------
def preprocess_features(X):
    selector = VarianceThreshold(threshold=1e-4)
    X_selected = selector.fit_transform(X)
    return X_selected

# -----------------------
# Rolling Ridge Regression (with train & test R²)
# -----------------------
def rolling_ridge(df, window=90):
    features = ['Open', 'High', 'Low', 'Close', 'Prev_Close',
                'MA3', 'MA7', 'MA14', 'Volatility', 'Momentum', 'RSI']
    
    X = df[features]
    y = df['Return']

    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=2, include_bias=False)

    preds = []
    actuals = []
    train_r2_list = []

    alphas = np.logspace(-3, 3, 10)

    for i in range(window, len(df)):
        X_train = X.iloc[i-window:i]
        y_train = y.iloc[i-window:i]
        X_test = X.iloc[i:i+1]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        X_train_pre = preprocess_features(X_train_poly)
        X_test_pre = X_test_poly[:, :X_train_pre.shape[1]]

        model = RidgeCV(alphas=alphas, store_cv_results=True)
        model.fit(X_train_pre, y_train)

        # Predict test
        y_pred = model.predict(X_test_pre)
        preds.append(y_pred[0])
        actuals.append(y.iloc[i])

        # Compute train R² for the rolling window
        y_train_pred = model.predict(X_train_pre)
        train_r2 = r2_score(y_train, y_train_pred)
        train_r2_list.append(train_r2)

    df_out = df.iloc[window:].copy()
    df_out['Predicted_Return'] = preds
    df_out['Actual_Return'] = actuals
    df_out['Train_R2'] = train_r2_list  # per rolling window train R²
    return df_out

# -----------------------
# Evaluation
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
# Plot Actual vs Predicted
# -----------------------
def plot_predictions(df, label):
    plt.figure(figsize=(10, 5))
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
# Main Runner for Streamlit
# -----------------------
def main(dataframes):
    results = []
    figures = []

    for label, df in dataframes.items():
        try:
            df = df[['Open', 'High', 'Low', 'Close']].copy()
            df = create_features(df)
            if len(df) < 100:
                print(f"⚠️ Skipping {label}: Not enough data points.")
                continue

            result_df = rolling_ridge(df)
            metrics = evaluate_results(result_df)
            fig = plot_predictions(result_df, label)

            results.append({
                'label': label,
                'train_r2': metrics['Train R² Score'],
                'test_r2': metrics['Test R² Score'],
                'test_mse': metrics['Test MSE'],
                'last_actual': metrics['Last Actual Return (%)'] / 100,
                'last_predicted': metrics['Last Predicted Return (%)'] / 100
            })

            figures.append(fig)

        except Exception as e:
            print(f"❌ Error processing {label}: {e}")
            continue

    # Compute portfolio average R² and MSE
    if results:
        avg_train_r2 = np.mean([r['train_r2'] for r in results])
        avg_test_r2 = np.mean([r['test_r2'] for r in results])
        avg_mse = np.mean([r['test_mse'] for r in results])
        results.insert(0, {
            'label': 'Portfolio (Average)',
            'train_r2': avg_train_r2,
            'test_r2': avg_test_r2,
            'test_mse': avg_mse,
            'last_actual': np.mean([r['last_actual'] for r in results]),
            'last_predicted': np.mean([r['last_predicted'] for r in results])
        })

    return {'results': results, 'figures': figures}
