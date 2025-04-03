
---

**`tasks/tasks.md`**

```markdown
# Project Tasks: Interactive Portfolio Risk Analyzer

## Phase 1: Setup & Core Structure

*   [x] Create project directory and initialize Git.
*   [x] Set up Python virtual environment (`venv`).
*   [x] Install initial dependencies (`streamlit`, `pandas`, `numpy`, `scipy`, `yfinance`, `plotly`).
*   [x] Create `requirements.txt`.
*   [x] Create initial `app.py`, `README.md`, `.cursorrules`, `.gitignore`.
*   [x] Create `tasks/tasks.md`, `docs/structure.md`, `docs/methodology.md`.

## Phase 2: Backend - Data Handling

*   [x] Implement `load_portfolio(uploaded_file)` function in `risk_calculator.py` (or `app.py`).
    *   [x] Read CSV using Pandas.
    *   [x] Validate columns ('Ticker', 'Value').
    *   [x] Calculate weights from values.
    *   [x] Handle potential file read/format errors.
*   [x] Implement `fetch_data(tickers, start_date, end_date)` function.
    *   [x] Use `yfinance.download`.
    *   [x] Select 'Adj Close'.
    *   [x] Handle API errors / missing tickers gracefully.
    *   [x] Basic data cleaning (e.g., check NaNs).
*   [x] Implement `calculate_returns(prices_df)` function.
    *   [x] Calculate daily log returns.
    *   [x] Handle initial NaN row.

## Phase 3: Backend - Risk Calculations

*   [x] Implement `calculate_portfolio_returns(asset_returns, weights)` function.
    *   [x] Use dot product for weighted returns.
*   [x] Implement `calculate_historical_var_cvar(portfolio_returns, confidence_level)` function.
    *   [x] Calculate VaR using `.quantile()`.
    *   [x] Calculate CVaR (Expected Shortfall).
*   [x] Implement `calculate_parametric_var_cvar(asset_returns, weights, confidence_level)` function.
    *   [x] Calculate daily mean vector and covariance matrix for assets.
    *   [x] Calculate daily portfolio mean and standard deviation.
    *   [x] Calculate VaR using `scipy.stats.norm.ppf`.
    *   [x] Calculate parametric CVaR.

## Phase 4: Backend - Visualizations

*   [x] Implement `plot_allocation(weights_df)` function using Plotly Pie chart.
*   [x] Implement `plot_histogram(portfolio_returns, hist_var, param_var)` function using Plotly Histogram.
    *   [x] Add vertical lines/shapes for VaR levels.
    *   [x] Add annotations for VaR values.
*   [x] (Optional) Implement `plot_rolling_var(...)` function using Plotly Line chart.

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