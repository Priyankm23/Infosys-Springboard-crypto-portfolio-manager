# 📈 Crypto Investment Manager

This project is a **crypto portfolio analysis system** built with Python and SQLite.  
It calculates **asset returns, portfolio returns, and risk** using different **weighting strategies**, stores results in a database, and compares **single asset vs portfolio performance** with visualizations and insights.

---

## 🚀 Features
1. **Data Storage**
   - Daily price data for BTC, ETH, and USDC stored in SQLite (`crypto.db`).
   - Two tables created:
     - **portfolio** → stores portfolio summary (name, created date, expected return, risk).
     - **portfolio_assets** → stores portfolio weights for each asset.

2. **Weighting Rules**
   Portfolio weights are assigned using three rules:
   - **Equal Weight** → All assets get the same weight.
   - **Price Weight** → Allocation proportional to latest price.
   - **Inverse Volatility Weight** → Safer assets (like USDC) get higher weight due to lower volatility.

   **Weights capping to max 50%**
   ex - weights = {"BTC": 0.7, "ETH": 0.2, "USDC": 0.1}
      cap = 0.5
      Step 1 (cap):
      BTC=0.5, ETH=0.2, USDC=0.1 → sum=0.8
      
      Step 2 (excess):
      excess=0.2
      
      Step 4–5 (redistribute):
      uncapped assets = ETH (0.2), USDC (0.1), total=0.3
      ETH gets (0.2/0.3)*0.2 = 0.1333 → ETH=0.3333
      USDC gets (0.1/0.3)*0.2 = 0.0667 → USDC=0.1667
      
      Step 6 (normalize):
      BTC=0.5, ETH≈0.333, USDC≈0.167 → sum=1
   
3. **Metrics**
   - **Percent Change** → daily return of each asset.
   - **Portfolio Return** → weighted return of all assets.
   - **Portfolio Risk** → standard deviation of portfolio returns.

4. **Parallel Execution**
   - Portfolio calculations for different rules are executed in parallel using `ThreadPoolExecutor`.
   - Results are stored in the database for later analysis.

5. **Comparison & Visualization**
   - Compare **single asset returns** vs **portfolio returns** (first 100 days).
   - Smooth data with rolling average for readability.
   - Plot returns side by side.

6. **Insights**
   - Automatic textual insights after plotting, showing **average return, risk, and interpretation** for each strategy.

---

## 📊 Insights 

### Single Asset Insights
- **BTC** → High return potential but very high volatility.  
- **ETH** → Moderate returns with lower volatility compared to BTC.  
- **USDC** → Very stable, near-zero returns, almost no volatility.  

### Portfolio Insights
- **Equal Weight Portfolio**  
  - Average return is moderate.  
  - Risk is reduced compared to BTC-only or ETH-only.  
  - ✅ Balanced diversification.  

- **Price Weight Portfolio**  
  - BTC dominates due to its higher price.  
  - Higher risk, but also higher potential return.  
  - ⚠️ Not ideal for risk-averse investors.  

- **Inverse Volatility Portfolio**  
  - Favors USDC because of its low volatility.  
  - Very low risk but returns are also minimal.  
  - ✅ Safe-haven style portfolio.  

- **Interpretation**
  - BTC and ETH show negative average returns and high volatility.  
  - USDC remains stable with ~0% return and negligible risk.  
  - ✅ Equal Portfolio smooths out risks, achieving **lower volatility (1.01)** than BTC or ETH alone, proving diversification     benefits.  

## 📈 Cumulative Insights

  - **BTC** → Strong growth in early days but large swings; ends with positive cumulative return but very high volatility.  
  - **ETH** → Similar growth path but slightly smoother than BTC; cumulative curve shows both upside and significant drawdowns.  
  - **USDC** → Flat cumulative line (~0%), confirming its role as a stability anchor.  
  - **Equal Portfolio** → Smoother upward curve; diversification dampens BTC/ETH volatility while still capturing upside.  
  
  **Sharpe Ratio Insights**  
  - BTC & ETH → Positive cumulative growth but low Sharpe ratios due to volatility.  
  - USDC → Very low Sharpe (near 0) since returns are ~0%.  
  - Equal Portfolio → Higher Sharpe than BTC/ETH, proving **better risk-adjusted performance**.  

  **Takeaway** → The cumulative analysis confirms that diversification improves long-term stability and risk-adjusted performance compared to holding a single volatile asset.

---

## 🛠️ Tech Stack
- **Python** → main logic
- **SQLite** → database
- **pandas, numpy** → data processing
- **matplotlib** → visualization
- **concurrent.futures** → parallel execution

---

