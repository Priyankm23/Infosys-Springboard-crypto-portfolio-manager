# Crypto Investment Manager

This project provides a comprehensive suite of tools for managing and analyzing cryptocurrency investments. It automates data collection, portfolio construction, risk assessment, and performance analysis, enabling users to make data-driven investment decisions.

## Core Features

- **Automated Data Pipeline**
  The system automatically ingests historical price data for cryptocurrencies like BTC, ETH, and USDC from CSV files into a centralized SQLite database. This creates a solid foundation for all subsequent analysis and modeling.

- **Strategic Portfolio Construction**
  It implements multiple rule-based investment strategies to build diversified portfolios. These strategies include:
    - **Equal Weighting**: Allocates capital equally among all assets.
    - **Price Weighting**: Allocates capital based on the relative price of each asset.
    - **Inverse Volatility**: Allocates more capital to less volatile assets to minimize risk.
    - **Sharpe Ratio-based**: Optimizes asset weights to maximize risk-adjusted returns.

- **Predictive Modeling for Returns**
  A machine learning model (Ridge Regression) is used to predict future asset returns. The model is trained on historical data and various technical indicators, such as moving averages and previous closing prices, to forecast performance for both individual assets and the entire portfolio.

- **Advanced Risk Management**
  The project includes a robust risk management module that continuously monitors the portfolio's health. It calculates key risk metrics like:
    - **Volatility**: The degree of variation of a trading price series over time.
    - **Sharpe & Sortino Ratios**: Measures of risk-adjusted return.
    - **Maximum Drawdown**: The maximum observed loss from a peak to a trough of a portfolio.
    - **Beta**: A measure of a portfolio's volatility in relation to the overall market.
  If any of these metrics breach predefined thresholds, an email alert is automatically sent to the user.

- **Performance and Stress Testing**
  The system backtests portfolio strategies against historical data and provides detailed performance analysis, including visualizations of returns. It also conducts stress tests to simulate how the portfolio would perform under various adverse market conditions (e.g., bull, bear, and high-volatility markets).

## Tech Stack

- **Language**: Python 3
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Database**: SQLite
- **Visualization**: Matplotlib

## Workflow

1.  **Initialization**: The database schema is created using `DB_portfolio.py`.
2.  **Data Loading**: Historical data is loaded from CSV files into the database via `data_handling.py`.
3.  **Metric Calculation**: `main.py` processes the raw data to compute technical indicators and stores them.
4.  **Strategy & Allocation**: `investment_rule.py` and `portfolio_math.py` work together to define investment strategies, calculate optimal asset weights, and construct the portfolio.
5.  **Prediction**: `predictor.py` runs the machine learning model to forecast future returns.
6.  **Monitoring**: `Risk_checker.py` runs periodically to assess portfolio risk and trigger alerts if necessary.

## Module Overview

| File                 | Description                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `main.py`            | Orchestrates the calculation of technical metrics (e.g., moving averages, volatility) from raw price data.  |
| `data_handling.py`   | A utility script for populating the SQLite database from source CSV files.                                |
| `DB_portfolio.py`    | Manages the database schema and provides functions for storing and retrieving portfolio data.               |
| `investment_rule.py` | Implements high-level investment strategies, fetches live prices, and runs market stress tests.             |
| `portfolio_math.py`  | Contains the core mathematical logic for portfolio weighting, backtesting, and performance analysis.        |
| `predictor.py`       | Implements the machine learning model to predict future returns based on historical features.             |
| `Risk_checker.py`    | Monitors portfolio risk metrics against predefined thresholds and sends email alerts upon violation.        |

## Getting Started

1.  **Prerequisites**: Ensure you have Python 3 and pip installed.
2.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd crypto-investment-manager
    ```
3.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
4.  **Setup the Database**:
    Run the `DB_portfolio.py` script to initialize the database and create the required tables.
    ```bash
    python DB_portfolio.py
    ```
5.  **Load Data**:
    Use `data_handling.py` to populate the database with the provided CSV data.
    ```bash
    python data_handling.py
    ```
6.  **Run the Modules**:
    Execute the different Python scripts to run the various parts of the application, starting with `main.py` to process the base data.
    ```bash
    python main.py
    python investment_rule.py
    python portfolio_math.py
    ```