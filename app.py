import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Import graph_objects for more control
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Portfolio Risk Analyzer")

# --- Title ---
st.title("Interactive Portfolio Risk Analyzer")
st.write("""
Analyze the risk of your stock portfolio using Historical Simulation
and Parametric (Variance-Covariance) Value at Risk (VaR) and
Conditional Value at Risk (CVaR) methods. Also view rolling VaR estimates.
""")

# --- Helper Function: Load Portfolio ---
def load_portfolio(uploaded_file):
    """
    Reads the uploaded CSV file, validates its structure, calculates weights,
    and returns a pandas DataFrame.

    Args:
        uploaded_file: The file object uploaded via st.file_uploader.

    Returns:
        pandas.DataFrame: DataFrame with 'Ticker', 'Value', and 'Weight' columns.

    Raises:
        ValueError: If the file is invalid (format, missing columns, bad data).
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    try:
        # Attempt to read the CSV
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = {'Ticker', 'Value'}
        if not required_cols.issubset(df.columns):
            missing_cols = required_cols - set(df.columns)
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        # Validate Ticker column (basic check for empty strings)
        if df['Ticker'].isnull().any() or (df['Ticker'] == '').any():
             raise ValueError("The 'Ticker' column contains missing or empty values.")
        # Ensure Tickers are strings
        df['Ticker'] = df['Ticker'].astype(str)

        # Validate and clean Value column
        # Attempt to convert to numeric, coercing errors (turns non-numeric into NaN)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        # Check if any non-numeric values were found (became NaN)
        if df['Value'].isnull().any():
            raise ValueError("The 'Value' column contains non-numeric entries. Please ensure all values are numbers.")

        # Check for negative values
        if (df['Value'] < 0).any():
            raise ValueError("The 'Value' column contains negative values. Please provide positive values.")

        # Check for zero total value
        total_value = df['Value'].sum()
        if total_value <= 0:
            # If individual values are zero, that's okay, but total must be > 0 for weights
            if (df['Value'] == 0).all():
                raise ValueError("All portfolio 'Value' entries are zero. Cannot calculate weights.")
            # Filter out zero-value rows as they won't contribute to weights
            df = df[df['Value'] > 0].copy()
            total_value = df['Value'].sum() # Recalculate total value
            if total_value <= 0: # Should not happen if previous check passed, but defensive
                 raise ValueError("Total portfolio value is zero or negative after filtering zero-value rows.")


        # Calculate weights
        df['Weight'] = df['Value'] / total_value

        return df[['Ticker', 'Value', 'Weight']]

    except ValueError as ve:
        # Re-raise specific ValueErrors for clarity
        raise ve
    except Exception as e:
        # Catch other potential errors during read/parse
        st.error(f"Error reading or processing CSV file: {e}")
        # Raise a ValueError to be caught by the main app logic
        raise ValueError(f"Could not process CSV file. Ensure it is a valid CSV. Error details: {e}")


# --- Helper Function: Fetch Data ---
def fetch_data(tickers, start_date, end_date):
    """
    Downloads historical adjusted closing prices ('Close') for a list of tickers.
    Uses 'Close' price as yfinance auto_adjust=True by default now.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (datetime.date): Start date for data fetching.
        end_date (datetime.date): End date for data fetching.

    Returns:
        pandas.DataFrame: DataFrame with adjusted close prices, indexed by date.

    Raises:
        ValueError: If data fetching fails or returns unexpected results.
        ConnectionError: If unable to connect to Yahoo Finance.
    """
    st.write(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    try:
        # Download data - yfinance auto_adjust=True is default
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if data.empty:
            raise ValueError("No data fetched. Check tickers and date range.")

        # Select only 'Close'. Handle single vs multiple tickers.
        if len(tickers) == 1:
            if 'Close' in data.columns:
                close_prices = data[['Close']]
                close_prices.columns = tickers # Rename column to ticker name
            else:
                raise ValueError("Could not find 'Close' data in downloaded results for single ticker.")
        elif isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
            close_prices.columns = close_prices.columns # Ensure columns are just tickers
        else:
             raise ValueError("Unexpected data format received from yfinance for multiple tickers.")

        # Check for columns that are entirely NaN (indicates failure for specific tickers)
        failed_tickers = close_prices.columns[close_prices.isnull().all()].tolist()
        if failed_tickers:
            st.warning(f"Could not fetch data for tickers: {', '.join(failed_tickers)}. They will be excluded.")
            close_prices = close_prices.drop(columns=failed_tickers)
            tickers[:] = [t for t in tickers if t not in failed_tickers] # Modify list in-place
            if close_prices.empty:
                 raise ValueError("No data successfully fetched for any provided ticker.")

        # Check for any remaining NaNs
        initial_nan_count = close_prices.isnull().sum().sum()
        if initial_nan_count > 0:
            #st.write(f"Found {initial_nan_count} missing data points. Applying forward fill...") # Less verbose
            close_prices = close_prices.ffill() # Forward fill missing values
            remaining_nan_count = close_prices.isnull().sum().sum()
            if remaining_nan_count > 0:
                st.warning(f"{remaining_nan_count} missing values remain after ffill (likely at start). Dropping affected rows.")
                close_prices = close_prices.dropna() # Drop rows with remaining NaNs

        if close_prices.empty:
             raise ValueError("Data became empty after handling missing values.")

        st.success("Data fetched successfully.")
        return close_prices

    except ValueError as ve:
        raise ve # Re-raise specific errors
    except Exception as e:
        st.error(f"Failed to download or process data from Yahoo Finance: {e}")
        raise ConnectionError(f"Could not connect or process data from Yahoo Finance: {e}")


# --- Helper Function: Calculate Returns ---
def calculate_returns(prices_df):
    """
    Calculates daily logarithmic returns from a DataFrame of prices.

    Args:
        prices_df (pandas.DataFrame): DataFrame of adjusted close prices.

    Returns:
        pandas.DataFrame: DataFrame of daily log returns.
    """
    if prices_df.empty or len(prices_df) < 2:
         raise ValueError("Price data is empty or insufficient to calculate returns.")
    # Calculate log returns: ln(P_t / P_{t-1})
    log_returns = np.log(prices_df / prices_df.shift(1))
    # Drop the first row which will be NaN after shift(1)
    log_returns = log_returns.dropna()
    if log_returns.empty:
        raise ValueError("Returns data became empty after dropping initial NaN row.")
    return log_returns


# --- Helper Function: Calculate Portfolio Returns ---
def calculate_portfolio_returns(asset_returns, weights):
    """ Calculates daily portfolio returns based on asset returns and weights. """
    # Ensure weights Series index matches asset_returns columns
    if not asset_returns.columns.equals(weights.index):
         raise ValueError("Asset return columns and weight indices do not match.")
    # Calculate weighted returns (dot product)
    portfolio_returns = asset_returns.dot(weights)
    portfolio_returns.name = "Portfolio Return" # Name the series
    return portfolio_returns


# --- Helper Function: Calculate Historical VaR/CVaR ---
def calculate_historical_var_cvar(portfolio_returns, confidence_level):
    """ Calculates Historical VaR and CVaR. """
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns data is empty.")
    # VaR is the quantile of the historical distribution
    alpha = 1 - confidence_level
    hist_var = portfolio_returns.quantile(alpha)
    # CVaR is the mean of returns less than or equal to VaR
    hist_cvar = portfolio_returns[portfolio_returns <= hist_var].mean()
    # Return as positive values representing potential loss
    return -hist_var, -hist_cvar


# --- Helper Function: Calculate Parametric VaR/CVaR ---
def calculate_parametric_var_cvar(asset_returns, weights, confidence_level):
    """ Calculates Parametric VaR and CVaR assuming normality. """
    if asset_returns.empty:
        raise ValueError("Asset returns data is empty.")
    if not asset_returns.columns.equals(weights.index):
         raise ValueError("Asset return columns and weight indices do not match.")

    # Calculate portfolio daily mean and standard deviation
    portfolio_mean_daily = asset_returns.dot(weights).mean()
    cov_matrix_daily = asset_returns.cov()
    portfolio_std_dev_daily = np.sqrt(weights.T @ cov_matrix_daily @ weights)

    if portfolio_std_dev_daily == 0:
        st.warning("Portfolio standard deviation is zero. Cannot calculate Parametric VaR/CVaR meaningfully.")
        return 0.0, 0.0

    # Calculate Z-score for the given confidence level (left tail)
    alpha = 1 - confidence_level
    z_score = norm.ppf(alpha)

    # Calculate Parametric VaR
    param_var = -(portfolio_mean_daily + z_score * portfolio_std_dev_daily)

    # Calculate Parametric CVaR (Expected Shortfall under normality)
    param_cvar = -(portfolio_mean_daily - portfolio_std_dev_daily * (norm.pdf(z_score) / alpha))

    return param_var, param_cvar


# --- Helper Function: Calculate Rolling VaR ---
def calculate_rolling_var(portfolio_returns, confidence_level, window_days):
    """
    Calculates rolling Historical and Parametric VaR.

    Args:
        portfolio_returns (pd.Series): Series of daily portfolio returns.
        confidence_level (float): Confidence level for VaR (e.g., 0.95).
        window_days (int): Rolling window size in days.

    Returns:
        pd.DataFrame: DataFrame containing rolling VaR series, or None if error.
    """
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("Portfolio returns data is empty, cannot calculate rolling VaR.")
        return None
    if len(portfolio_returns) < window_days:
        st.warning(f"Lookback period ({len(portfolio_returns)} days) is shorter than rolling window ({window_days} days). Cannot calculate rolling VaR.")
        return None

    alpha = 1 - confidence_level
    z_score = norm.ppf(alpha) # Z-score for parametric calculation

    # Calculate Rolling Historical VaR
    rolling_hist_var = portfolio_returns.rolling(window=window_days).quantile(alpha)

    # Calculate Rolling Parametric VaR (assuming zero mean for daily returns)
    rolling_std_dev = portfolio_returns.rolling(window=window_days).std()
    rolling_param_var = z_score * rolling_std_dev # Z * sigma (mean assumed zero)

    # Combine into a DataFrame - multiply by -1 to represent positive loss values
    rolling_var_df = pd.DataFrame({
        'Historical VaR': -rolling_hist_var,
        'Parametric VaR': -rolling_param_var
    })

    # Drop initial NaN rows resulting from the rolling window
    rolling_var_df = rolling_var_df.dropna()

    if rolling_var_df.empty:
        st.warning("Rolling VaR calculation resulted in an empty DataFrame after dropping NaNs.")
        return None

    return rolling_var_df


# --- Visualization Function: Plot Allocation ---
def plot_allocation(portfolio_df):
    """ Creates a Plotly Pie chart for portfolio allocation. """
    if portfolio_df is None or portfolio_df.empty:
        st.warning("Portfolio data is not available for allocation plot.")
        return None

    fig = px.pie(
        portfolio_df,
        names='Ticker',
        values='Weight',
        title='Portfolio Allocation by Weight',
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    return fig


# --- Visualization Function: Plot Histogram ---
def plot_histogram(portfolio_returns, hist_var, param_var, confidence_level_pct):
    """ Creates a Plotly Histogram of portfolio returns with VaR lines. """
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("Portfolio returns are not available for histogram plot.")
        return None

    fig = px.histogram(
        portfolio_returns,
        nbins=50,
        title='Distribution of Historical Daily Portfolio Returns',
        labels={'value': 'Daily Log Return', 'count': 'Frequency'},
        template='plotly_white'
    )
    fig.update_layout(showlegend=True, yaxis_title="Frequency", xaxis_title="Daily Log Return (%)")

    # Format VaR values for labels
    hist_var_label = f"Hist. VaR {confidence_level_pct}% ({hist_var:.2%})"
    param_var_label = f"Param. VaR {confidence_level_pct}% ({param_var:.2%})"

    fig.add_vline(
        x=-hist_var, line_dash="dash", line_color="red",
        annotation_text=hist_var_label,
        annotation_position="top left"
    )
    fig.add_vline(
        x=-param_var, line_dash="dot", line_color="blue",
        annotation_text=param_var_label,
        annotation_position="bottom left"
    )

    fig.update_xaxes(tickformat=".2%") # Format x-axis as percentage
    return fig


# --- Visualization Function: Plot Rolling VaR ---
def plot_rolling_var(rolling_var_df, confidence_level_pct):
    """ Creates a Plotly Line chart for rolling VaR estimates. """
    if rolling_var_df is None or rolling_var_df.empty:
        st.warning("Rolling VaR data is not available for plotting.")
        return None

    # Try to infer window size for title, default if fails
    try:
        window_size_days = int(rolling_var_df.index.to_series().diff().median().days)
        title_window = f"{window_size_days}-Day"
    except:
        title_window = "X-Day" # Fallback title


    fig = px.line(
        rolling_var_df * 100, # Convert to percentage for plotting
        title=f'Rolling {title_window} VaR ({confidence_level_pct}%) Estimation',
        labels={'value': 'Estimated VaR (% of Portfolio)', 'variable': 'VaR Method', 'index': 'Date'},
        template='plotly_white'
    )

    fig.update_layout(
        yaxis_title="Estimated VaR (% of Portfolio)",
        xaxis_title="Date",
        legend_title="VaR Method",
        hovermode="x unified" # Improves hover tooltip
    )
    fig.update_yaxes(ticksuffix="%") # Format y-axis as percentage

    return fig


# --- Sidebar Inputs ---
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload Portfolio CSV", type="csv")
confidence_level_pct = st.sidebar.slider(
    "Confidence Level (%)", min_value=90.0, max_value=99.9, value=95.0, step=0.1
)
lookback_years = st.sidebar.number_input(
    "Lookback Period (Years)", min_value=1, max_value=10, value=2, step=1
)
rolling_window_days = st.sidebar.number_input(
    "Rolling VaR Window (Trading Days)",
    min_value=10, max_value=252, value=60, step=10
)
calculate_button = st.sidebar.button("Calculate Risk", type="primary")

# --- Main Application Logic ---
# Initialize variables
portfolio_df = None
adj_close_prices = None
asset_returns = None
portfolio_returns = None
hist_var, hist_cvar, param_var, param_cvar = (None,) * 4
rolling_var_df = None

if uploaded_file is not None and calculate_button:
    st.header("Analysis Results")
    st.info(f"""
        Processing portfolio from '{uploaded_file.name}'
        with {confidence_level_pct}% confidence level
        over a {lookback_years} year lookback period
        (Rolling VaR window: {rolling_window_days} days).
    """)

    try:
        # --- Step 1: Load Portfolio ---
        portfolio_df = load_portfolio(uploaded_file)
        tickers = portfolio_df['Ticker'].tolist()
        weights = portfolio_df.set_index('Ticker')['Weight'] # Set index before accessing

        # --- Step 2: Calculate Date Range ---
        end_date = datetime.date.today()
        # Fetch slightly more data to ensure enough for rolling window at the start
        buffer_days = rolling_window_days + 5 # Add a small buffer
        start_date = end_date - relativedelta(years=lookback_years) - datetime.timedelta(days=buffer_days)

        # --- Step 3: Fetch Historical Data ---
        adj_close_prices = fetch_data(tickers, start_date, end_date)

        # Check/update weights if tickers dropped
        if len(tickers) != len(weights):
            st.warning("Some tickers dropped. Recalculating weights.")
            portfolio_df = portfolio_df[portfolio_df['Ticker'].isin(tickers)].copy()
            total_value_remaining = portfolio_df['Value'].sum()
            if total_value_remaining <= 0: raise ValueError("Total value zero after removing tickers.")
            portfolio_df['Weight'] = portfolio_df['Value'] / total_value_remaining
            weights = portfolio_df.set_index('Ticker')['Weight']

        # Display portfolio summary (potentially updated)
        st.subheader("Portfolio Summary")
        st.dataframe(portfolio_df.style.format({'Value': '{:,.2f}', 'Weight': '{:.2%}'}))

        # Align weights and prices
        weights = weights.reindex(adj_close_prices.columns)
        if weights.isnull().any():
             missing_in_prices = weights[weights.isnull()].index.tolist()
             st.warning(f"Could not find price data columns matching weights for: {missing_in_prices}. Removing.")
             weights = weights.dropna()
             adj_close_prices = adj_close_prices[weights.index] # Adjust prices df *before* return calculation
             if weights.sum() <= 0: raise ValueError("Weights sum to zero or less after removing assets.")
             weights = weights / weights.sum() # Renormalize

        # --- Step 4: Calculate Returns (Asset) ---
        asset_returns = calculate_returns(adj_close_prices)

        # --- Step 5: Calculate Portfolio Returns ---
        # Make sure weights match the final asset_returns columns after potential filtering
        weights = weights.reindex(asset_returns.columns)
        if weights.isnull().any(): raise ValueError("Weight mismatch after return calculation.") # Should not happen now
        weights = weights / weights.sum() # Ensure re-normalized if needed
        portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

        # --- Step 6: Calculate Risk Metrics (Static VaR/CVaR) ---
        # Use returns from the period matching the user's lookback choice for static VaR
        lookback_start_date_strict = end_date - relativedelta(years=lookback_years)
        static_portfolio_returns = portfolio_returns[portfolio_returns.index >= pd.Timestamp(lookback_start_date_strict)]
        static_asset_returns = asset_returns[asset_returns.index >= pd.Timestamp(lookback_start_date_strict)]

        if static_portfolio_returns.empty or static_asset_returns.empty:
            raise ValueError("Not enough data within the specified lookback period for static VaR/CVaR calculation after initial processing.")

        confidence_level = confidence_level_pct / 100.0
        hist_var, hist_cvar = calculate_historical_var_cvar(static_portfolio_returns, confidence_level)
        param_var, param_cvar = calculate_parametric_var_cvar(static_asset_returns, weights, confidence_level) # Weights based on full portfolio

        # --- Step 7: Calculate Rolling VaR ---
        # Use the full portfolio_returns series which includes the buffer period
        rolling_var_df = calculate_rolling_var(portfolio_returns, confidence_level, rolling_window_days)


        # --- Display Calculated Results (Static) ---
        st.subheader("Risk Metrics (Daily % of Portfolio)")
        st.warning("""
            **Note:** VaR/CVaR are expressed as a percentage loss potential...
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Historical VaR ({confidence_level_pct}%)", f"{hist_var:.2%}")
            st.metric(f"Historical CVaR ({confidence_level_pct}%)", f"{hist_cvar:.2%}")
        with col2:
            st.metric(f"Parametric VaR ({confidence_level_pct}%)", f"{param_var:.2%}")
            st.metric(f"Parametric CVaR ({confidence_level_pct}%)", f"{param_cvar:.2%}")

        # --- Display Visualizations ---
        st.subheader("Visualizations")

        # Allocation Plot
        fig_pie = plot_allocation(portfolio_df)
        if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
        else: st.write("Could not generate allocation plot.")

        # Histogram Plot (use static returns for the histogram period)
        fig_hist = plot_histogram(static_portfolio_returns, hist_var, param_var, confidence_level_pct)
        if fig_hist: st.plotly_chart(fig_hist, use_container_width=True)
        else: st.write("Could not generate histogram plot.")

        # Rolling VaR Plot
        fig_rolling_var = plot_rolling_var(rolling_var_df, confidence_level_pct)
        if fig_rolling_var:
            st.plotly_chart(fig_rolling_var, use_container_width=True)
        else:
            st.write("Could not generate rolling VaR plot (check warnings above).")


    except (ValueError, ConnectionError) as ve:
        st.error(f"Data or Calculation Error: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis:")
        st.exception(e) # Display full traceback for debugging
        st.stop()

elif calculate_button and uploaded_file is None:
    st.warning("⚠️ Please upload a portfolio CSV file first.")
else:
    st.info("⬆️ Upload your portfolio CSV and set parameters in the sidebar to begin analysis.")

# --- Footer/Info in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.info("""
**Disclaimer:** This application is for educational purposes only.
Financial decisions should not be made based solely on the results provided here.
Risk metrics are estimates based on historical data and model assumptions.
""")