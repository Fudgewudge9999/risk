
---

**`tasks/tasks.md`**

```markdown
# Project Tasks: Interactive Portfolio Risk Analyzer

## Phase 1: Setup & Core Structure

*   [ ] Create project directory and initialize Git.
*   [ ] Set up Python virtual environment (`venv`).
*   [ ] Install initial dependencies (`streamlit`, `pandas`, `numpy`, `scipy`, `yfinance`, `plotly`).
*   [ ] Create `requirements.txt`.
*   [ ] Create initial `app.py`, `README.md`, `.cursorrules`, `.gitignore`.
*   [ ] Create `tasks/tasks.md`, `docs/structure.md`, `docs/methodology.md`.

## Phase 2: Backend - Data Handling

*   [ ] Implement `load_portfolio(uploaded_file)` function in `risk_calculator.py` (or `app.py`).
    *   [ ] Read CSV using Pandas.
    *   [ ] Validate columns ('Ticker', 'Value').
    *   [ ] Calculate weights from values.
    *   [ ] Handle potential file read/format errors.
*   [ ] Implement `fetch_data(tickers, start_date, end_date)` function.
    *   [ ] Use `yfinance.download`.
    *   [ ] Select 'Adj Close'.
    *   [ ] Handle API errors / missing tickers gracefully.
    *   [ ] Basic data cleaning (e.g., check NaNs).
*   [ ] Implement `calculate_returns(prices_df)` function.
    *   [ ] Calculate daily log returns.
    *   [ ] Handle initial NaN row.

## Phase 3: Backend - Risk Calculations

*   [ ] Implement `calculate_portfolio_returns(asset_returns, weights)` function.
    *   [ ] Use dot product for weighted returns.
*   [ ] Implement `calculate_historical_var_cvar(portfolio_returns, confidence_level)` function.
    *   [ ] Calculate VaR using `.quantile()`.
    *   [ ] Calculate CVaR (Expected Shortfall).
*   [ ] Implement `calculate_parametric_var_cvar(asset_returns, weights, confidence_level)` function.
    *   [ ] Calculate daily mean vector and covariance matrix for assets.
    *   [ ] Calculate daily portfolio mean and standard deviation.
    *   [ ] Calculate VaR using `scipy.stats.norm.ppf`.
    *   [ ] Calculate parametric CVaR.

## Phase 4: Backend - Visualizations

*   [ ] Implement `plot_allocation(weights_df)` function using Plotly Pie chart.
*   [ ] Implement `plot_histogram(portfolio_returns, hist_var, param_var)` function using Plotly Histogram.
    *   [ ] Add vertical lines/shapes for VaR levels.
    *   [ ] Add annotations for VaR values.
*   [ ] (Optional) Implement `plot_rolling_var(...)` function using Plotly Line chart.

## Phase 5: Frontend - Streamlit App (`app.py`)

*   [ ] Set up basic Streamlit page layout (title, sidebar).
*   [ ] Add file uploader widget (`st.file_uploader`) to sidebar.
*   [ ] Add widgets for confidence level (`st.slider` or `st.selectbox`) and lookback period (`st.number_input`) to sidebar.
*   [ ] Add a button (`st.button`) to trigger calculations.
*   [ ] Implement main app logic:
    *   [ ] Check for uploaded file and button press.
    *   [ ] Call backend functions sequentially.
    *   [ ] Handle errors from backend calls and display using `st.error`.
*   [ ] Display results:
    *   [ ] Use `st.metric` for key VaR/CVaR figures.
    *   [ ] Use `st.plotly_chart` to show plots.
    *   [ ] Use `st.dataframe` or `st.table` for portfolio weights summary.
    *   [ ] Use `st.tabs` or `st.expander` to organize output.

## Phase 6: Testing, Refinement & Documentation

*   [ ] Perform manual testing with sample valid/invalid CSVs.
*   [ ] Verify calculations with a simple known portfolio.
*   [ ] Refactor code for clarity and modularity (e.g., ensure `risk_calculator.py` is well-defined if used).
*   [ ] Add comments and docstrings to functions.
*   [ ] Add more robust error handling throughout.
*   [ ] Update `README.md` with final details, screenshot.
*   [ ] Finalize `requirements.txt`.
*   [ ] Ensure Git history is clean.