# Risk Calculation Methodologies

This document details the specific methods used to calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) in this project. All calculations are typically performed on daily returns unless otherwise specified.

## 1. Historical Simulation

This non-parametric method directly uses the distribution of past portfolio returns.

**Assumptions:**
*   The historical distribution of returns over the lookback period is a good representation of the distribution of returns in the near future.
*   Does not assume normality.

**Calculation Steps:**

1.  **Calculate Asset Returns:** Obtain daily log returns for each asset in the portfolio over the specified lookback period.
    `Log Return = ln(Price_t / Price_{t-1})`
2.  **Calculate Portfolio Returns:** Compute the weighted average of the asset returns for each day, based on the initial portfolio weights.
    `Portfolio Return_t = Σ (Weight_i * Asset Return_{i,t})`
3.  **Sort Portfolio Returns:** Arrange the historical daily portfolio returns in ascending order.
4.  **Calculate VaR:** Find the return value at the `(1 - confidence_level)` percentile of the sorted historical distribution. For example, for 95% confidence (alpha=0.95), find the 5th percentile (quantile `q = 0.05`).
    `VaR = - portfolio_returns.quantile(1 - confidence_level)`
    *(Note: VaR is typically reported as a positive value representing a loss.)*
5.  **Calculate CVaR (Expected Shortfall):** Calculate the average of all portfolio returns that are less than or equal to the VaR value calculated in step 4.
    `CVaR = - portfolio_returns[portfolio_returns <= -VaR].mean()`
    *(Note: CVaR is also reported as a positive value representing the expected loss given that the loss exceeds VaR.)*

## 2. Parametric (Variance-Covariance) Method

This method assumes that portfolio returns follow a specific distribution, typically the normal distribution.

**Assumptions:**
*   Daily portfolio returns are normally distributed.
*   The mean (μ) and standard deviation (σ) calculated from the historical data are good estimates for the near future.

**Calculation Steps:**

1.  **Calculate Asset Returns:** Obtain daily log returns for each asset over the lookback period.
2.  **Calculate Asset Statistics:**
    *   Calculate the *mean vector* (μ_asset) of daily returns for each asset.
    *   Calculate the *covariance matrix* (Σ) of daily returns between all assets.
3.  **Calculate Portfolio Statistics (Daily):**
    *   Calculate the expected portfolio daily mean return:
        `μ_portfolio = weights^T * μ_asset`
    *   Calculate the portfolio daily variance:
        `σ²_portfolio = weights^T * Σ * weights`
    *   Calculate the portfolio daily standard deviation:
        `σ_portfolio = sqrt(σ²_portfolio)`
4.  **Find Z-score:** Determine the Z-score corresponding to the desired confidence level using the inverse of the standard normal cumulative distribution function (CDF).
    `Z = scipy.stats.norm.ppf(confidence_level)`
    *(Note: For VaR, we are interested in the left tail, so often `scipy.stats.norm.ppf(1 - confidence_level)` is used, resulting in a negative Z-score for typical confidence levels > 50%). Let's use `alpha = 1 - confidence_level` for clarity.*
    `Z_alpha = scipy.stats.norm.ppf(alpha)` (This will be negative)
5.  **Calculate VaR:** Combine the portfolio statistics and the Z-score.
    `VaR = -(μ_portfolio + Z_alpha * σ_portfolio)`
    *(Often, for short horizons like 1-day VaR, the mean μ_portfolio is assumed to be zero: `VaR ≈ -Z_alpha * σ_portfolio`)*
    *(Note: Ensure μ and σ are for the same period, e.g., daily. Result is reported as positive loss.)*
6.  **Calculate CVaR (Assuming Normality):** The formula for CVaR under normality is:
    `CVaR = -(μ_portfolio - σ_portfolio * [pdf(Z_alpha) / alpha])`
    where `pdf(Z_alpha)` is the value of the standard normal probability density function at `Z_alpha`, and `alpha = 1 - confidence_level`.
    `CVaR = -(μ_portfolio - σ_portfolio * (scipy.stats.norm.pdf(Z_alpha) / (1 - confidence_level)))`
    *(Result reported as positive loss.)*

---

Remember to populate the `fixes/` directory only if you encounter complex bugs needing detailed write-ups during development. These initial files provide a solid documentation foundation based on your `.cursorrules`.