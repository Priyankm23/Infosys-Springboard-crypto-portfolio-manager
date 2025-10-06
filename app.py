import streamlit as st
import pandas as pd
import sqlite3

# Import refactored functions
from DB_portfolio import init_db
from portfolio_math import run_and_plot_strategy
from predictor import main as run_predictor
from investment_rule import run_investment_strategy
from Risk_checker import run_risk_check
from main import run_metric_calculation


# ------------------ DATABASE EXPLORER ------------------ #
def view_raw_data():
    conn = sqlite3.connect("db/crypto.db")
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    
    table_name = st.selectbox("Select a table to view", tables['name'])
    if table_name:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Could not read table: {e}")
    
    conn.close()


# ------------------ METRIC CALCULATION ------------------ #
def calculate_metrics():
    if st.button("Run Metric Calculation"):
        with st.spinner("Calculating metrics for BTC and ETH..."):
            output = run_metric_calculation()
            st.success("Metric calculation and database insertion complete!")

            st.subheader("Database Insertion Log")
            st.text_area("Log", output["message"], height=100)

            st.subheader("Calculated Metrics")
            for symbol, df in output["dataframes"].items():
                st.markdown(f"**{symbol} Metrics**")
                st.dataframe(df)


# ------------------ PORTFOLIO ANALYSIS ------------------ #
def run_portfolio_analysis_page():
    rule_name = st.selectbox("Select a portfolio weighting rule", ["Equal", "Price", "InvVol"])

    if st.button("Analyze Portfolio"):
        with st.spinner(f"Running analysis for {rule_name} weighted portfolio..."):
            fig1, fig2, insights = run_and_plot_strategy(rule_name)
            st.success("Analysis complete!")

            st.subheader("Performance vs. Assets (15-Day Returns)")
            st.pyplot(fig1)

            st.subheader("Performance vs. Assets (Cumulative Returns)")
            st.pyplot(fig2)

            st.subheader("Insights")
            st.text(insights)


# ------------------ INVESTMENT STRATEGY ------------------ #
def run_investment_strategy_page():
    st.header("Sharpe-based Investment Strategy")

    if st.button("Run Strategy & Update Weights"):
        with st.spinner("Fetching live data and calculating optimal weights..."):
            output = run_investment_strategy()
            st.success("Strategy executed successfully!")

            st.subheader("Calculated Optimal Weights")
            st.json(output["weights"])

            st.subheader("Expected Portfolio Return (Latest Day)")
            st.write(f"{output['portfolio_return']:.4%}")

            st.subheader("Stress Test Results")
            st.json(output["stress_test_results"])

            st.subheader("Insights")
            for insight in output["insights"]:
                st.write(f"- {insight}")


# ------------------ PREDICTIONS ------------------ #
def run_predictions_page():
    st.header("Return Prediction with Regression Model")

    if st.button("Run Prediction Models"):
        with st.spinner("Training models and making predictions..."):
            output = run_predictor()
            st.success("Prediction models ran successfully!")

            st.subheader("Prediction Results")
            for res in output['results']:
                st.markdown(f"**{res['label']}**")

                col1, col2 = st.columns(2)
                col1.metric("Test R² Score", f"{res['test_r2']:.4f}")
                col2.metric("Test MSE", f"{res['test_mse']:.6f}")
                col1.metric("Last Actual Return", f"{res['last_actual']:.4%}")
                col2.metric("Last Predicted Return", f"{res['last_predicted']:.4%}")

            st.subheader("Actual vs. Predicted Plots")
            for fig in output['figures']:
                st.pyplot(fig)


# ------------------ RISK CHECKER ------------------ #
def check_risk_page():
    st.header("Run Portfolio Risk Check")

    if st.button("Check for Risk Violations"):
        with st.spinner("Computing risk metrics..."):
            metrics, alert_message = run_risk_check()
            st.success("Risk check complete!")

            st.subheader("Current Portfolio Risk Metrics")
            cols = st.columns(len(metrics))

            for i, (k, v) in enumerate(metrics.items()):
                if k in ["volatility", "max_drawdown", "max_weight"]:
                    cols[i].metric(k.replace("_", " ").title(), f"{v:.2%}")
                else:
                    cols[i].metric(k.replace("_", " ").title(), f"{v:.2f}")

            if alert_message:
                st.subheader("⚠️ Risk Alert Triggered ⚠️")
                st.warning(alert_message)
            else:
                st.subheader("✅ No Risk Alerts")
                st.success("Portfolio is within all defined risk thresholds.")


# ------------------ MAIN APP ------------------ #
def main():
    st.set_page_config(page_title="Crypto Investment Manager", layout="wide")
    st.sidebar.title("Crypto Investment Manager")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Calculate Technical Metrics",
            "Run Investment Strategy",
            "Portfolio Performance Analysis",
            "Predict Future Returns",
            "Check Portfolio Risk",
            "Database Explorer"
        ]
    )

    st.header(page)

    if page == "Database Explorer":
        view_raw_data()
    elif page == "Calculate Technical Metrics":
        calculate_metrics()
    elif page == "Portfolio Performance Analysis":
        run_portfolio_analysis_page()
    elif page == "Run Investment Strategy":
        run_investment_strategy_page()
    elif page == "Predict Future Returns":
        run_predictions_page()
    elif page == "Check Portfolio Risk":
        check_risk_page()


# ------------------ ENTRY POINT ------------------ #
if __name__ == "__main__":
    try:
        init_db()
    except Exception as e:
        # This can happen if the tables already exist, which is fine.
        print(f"Database already initialized or error: {e}")

    main()
