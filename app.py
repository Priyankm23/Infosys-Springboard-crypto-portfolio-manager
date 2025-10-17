import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

# Import refactored functions
from DB_portfolio import init_db
from portfolio_math import run_and_plot_strategy
from predictor import main as run_predictor
from investment_rule import run_investment_strategy
from Risk_checker import run_risk_check
from main import run_metric_calculation
from auth import init_user_db, add_user, check_user

# ------------------ AUTHENTICATION ------------------ #

def login_page():
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_user(email, password):
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid email or password")

def signup_page():
    st.header("Sign Up")
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if add_user(name, email, password):
            st.success("User created successfully! Please login.")
        else:
            st.error("Email already exists.")


# ------------------ METRIC CALCULATION ------------------ #
def calculate_metrics():
    if st.button("Run Metric Calculation"):
        if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
            with st.spinner("Calculating metrics for uploaded data..."):
                output = run_metric_calculation(uploaded_data=st.session_state.uploaded_data)
                st.success("Metric calculation complete!")

                st.subheader("Calculated Metrics")
                for symbol, df in output["dataframes"].items():
                    st.markdown(f"**{symbol} Metrics**")
                    st.dataframe(df)
        else:
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
            fig1, fig2, insights, weights = run_and_plot_strategy(rule_name, st.session_state.uploaded_data)
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
        if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
            with st.spinner("Calculating investment strategy for uploaded data..."):
                output = run_investment_strategy(st.session_state.uploaded_data)
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
        else:
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
    st.header("📈 Return Prediction with Regression Model")

    uploaded_data = st.session_state.uploaded_data

    if not uploaded_data:
        st.warning("Please upload at least one cryptocurrency CSV file using the sidebar uploader.")
        return

    if st.button("Run Prediction Models"):
        with st.spinner("Training models and making predictions..."):
            try:
                output = run_predictor(uploaded_data)
                st.success("✅ Prediction models ran successfully!")

                st.subheader("📊 Prediction Results")

                for res in output['results']:
                    st.markdown(f"### {res['label']}")

                    # safely format numeric values
                    def safe_format(value, fmt):
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            return "N/A"
                        return format(value, fmt)

                    test_r2_str = safe_format(res.get('test_r2'), ".4f")
                    train_r2_str = safe_format(res.get('train_r2'), ".4f")
                    test_mse_str = safe_format(res.get('test_mse'), ".6f")

                    last_actual_pct = res.get('last_actual', 0.0)
                    last_predicted_pct = res.get('last_predicted', 0.0)
                    next_day_predicted_pct = res.get('next_day_predicted', 0.0)  # 🔹 NEW

                    # --- Display Returns (Bigger Font) ---
                    st.markdown(
                        f"""
                        <div style='font-size:20px; font-weight:700; margin-bottom:8px;'>
                            Last Actual Return: <span style='color:#0CED88;'>{last_actual_pct:.4%}</span><br>
                            Last Predicted Return: <span style='color:#0c84ed;'>{last_predicted_pct:.4%}</span><br>
                            Next Day Predicted Return: <span style='color:#0CBCED;'>{next_day_predicted_pct:.4%}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # --- Display Scores (Smaller Font) ---
                    st.markdown(
                        f"""
                        <div style='font-size:13px; color:#F3E736; margin-bottom:10px;'>
                            <b>Scores:</b>&nbsp;
                            Test R² = {test_r2_str} &nbsp;|&nbsp;
                            Train R² = {train_r2_str} &nbsp;|&nbsp;
                            Test MSE = {test_mse_str}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --- Display Actual vs Predicted Plots ---
                st.subheader("📉 Actual vs. Predicted Plots")
                for fig in output.get('figures', []):
                    if fig is not None:
                        st.pyplot(fig)

                # --- Disclaimer ---
                st.markdown(
                    """
                    <hr>
                    <div style='font-size:14px; color:#444; background-color:#fff8f0; padding:12px; border-radius:8px;'>
                        ⚠️ <b>Disclaimer:</b> These predictions are computed by statistical models using historical price data.
                        They do not consider news, macro events, liquidity, or other market factors. 
                        <b>Do not rely solely on these outputs for trading decisions.</b>
                        Use them for research and educational purposes only.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ------------------ RISK CHECKER ------------------ #
def check_risk_page():
    st.header("Run Portfolio Risk Check")

    if st.button("Check for Risk Violations"):
        with st.spinner("Computing risk metrics..."):
            metrics, alert_message = run_risk_check(uploaded_data=st.session_state.uploaded_data)
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
    st.set_page_config(
    page_title="Crypto Investment Manager", 
    layout="wide",
    initial_sidebar_state="expanded",
    
)
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.sidebar.title("Authentication")
        auth_choice = st.sidebar.radio("Choose an action", ["Login", "Sign Up"])
        if auth_choice == "Login":
            login_page()
        else:
            signup_page()
    else:
        st.sidebar.title("Crypto Portfolio Manager")
        
        uploaded_files = st.sidebar.file_uploader("Upload your own CSV files", accept_multiple_files=True, type=['csv'])
        if uploaded_files:
            st.session_state.uploaded_data = {file.name.split('.')[0]: pd.read_csv(file) for file in uploaded_files}
        else:
            st.session_state.uploaded_data = None

        page = st.sidebar.radio(
            "Navigation",
            [
                "Calculate Technical Metrics",
                "Run Investment Strategy",
                "Portfolio Performance Analysis",
                "Predict Future Returns",
                "Check Portfolio Risk",
            ]
        )

        st.header(page)

        if page == "Calculate Technical Metrics":
            calculate_metrics()
        elif page == "Portfolio Performance Analysis":
            run_portfolio_analysis_page()
        elif page == "Run Investment Strategy":
            run_investment_strategy_page()
        elif page == "Predict Future Returns":
            run_predictions_page()
        elif page == "Check Portfolio Risk":
            check_risk_page()
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()


# ------------------ ENTRY POINT ------------------ #
if __name__ == "__main__":
    try:
        init_db()
        init_user_db()
    except Exception as e:
        # This can happen if the tables already exist, which is fine.
        print(f"Database already initialized or error: {e}")

    main()