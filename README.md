# Interactive Portfolio Risk Analyzer

## 📜 Overview

This project is a web application built with Python and Streamlit to analyze the risk of a user-provided stock portfolio. It calculates key risk metrics like Value at Risk (VaR) and Conditional Value at Risk (CVaR) using standard financial methodologies. The goal is to provide a simple, interactive tool for basic portfolio risk assessment.

## ✨ Features

*   **Portfolio Upload:** Upload a portfolio via CSV file (specifying Tickers and Values).
*   **Customizable Parameters:** Select confidence levels (e.g., 95%, 99%) and historical data lookback periods.
*   **Risk Metrics:** Calculates VaR and CVaR using:
    *   **Historical Simulation:** Based directly on past portfolio performance.
    *   **Parametric (Variance-Covariance):** Assuming normally distributed returns.
*   **Interactive Visualizations:**
    *   Portfolio allocation pie chart.
    *   Histogram of historical portfolio returns with VaR levels marked.
    *   (Optional) Rolling VaR plot over time.
*   **Web-Based UI:** Built with Streamlit for easy interaction.

## ⚠️ Limitations

*   **Asset Scope:** Primarily supports US stocks available on Yahoo Finance.
*   **Data Source:** Dependent on `yfinance` data quality and availability.
*   **Model Assumptions:** Historical simulation assumes the past predicts the future; Parametric method assumes return normality (which often doesn't hold).
*   **Static Weights:** Calculations assume fixed portfolio weights over the lookback period.
*   **Exclusions:** Does not account for transaction costs, taxes, dividends (unless included in Adjusted Close), or factor risks.
*   **No Backtesting:** Does not include formal statistical validation of the VaR models.

## 🚀 Technology Stack

*   **Python 3.8+**
*   **Streamlit:** Web application framework
*   **Pandas:** Data manipulation and analysis
*   **NumPy:** Numerical computations
*   **SciPy:** Statistical functions (specifically `scipy.stats.norm`)
*   **yfinance:** Downloading historical stock data
*   **Plotly:** Interactive visualizations

## 🛠️ Setup & Installation

1.  **Clone the repository (or create project directory):**
    ```bash
    # git clone <your-repo-url> # If using Git
    # cd interactive-portfolio-risk-analyzer
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Running the App

Ensure your virtual environment is activated. Run the Streamlit app from your terminal:

```bash
streamlit run app.py