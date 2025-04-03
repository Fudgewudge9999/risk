# Project Structure

This document outlines the basic file and directory structure for the Interactive Portfolio Risk Analyzer.

*   **`.cursorrules`**: Rules and guidelines for development workflow, especially when using AI assistance (Cursor IDE).
*   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (e.g., `venv/`, `__pycache__/`, `.env`).
*   **`app.py`**: The main Streamlit application script. Contains the UI definition (widgets, layout) and orchestrates the calls to backend calculation functions.
*   **`risk_calculator.py`** (Optional but Recommended): A separate Python module containing the core financial logic:
    *   Data fetching (`yfinance`).
    *   Return calculations.
    *   VaR/CVaR implementations (Historical, Parametric).
    *   Plotting functions (generating Plotly figures).
    *   This promotes separation of concerns between the UI and the calculations.
*   **`requirements.txt`**: Lists all Python package dependencies required to run the project. Generated via `pip freeze`.
*   **`README.md`**: The primary documentation file. Contains project overview, setup, usage instructions, features, limitations, etc. **Start here!**
*   **`tasks/`**: Directory for planning and tracking development work.
    *   **`tasks.md`**: Detailed breakdown of features and implementation steps using Markdown task lists.
*   **`docs/`**: Directory for supplementary documentation.
    *   **`structure.md`**: This file - explains the project layout.
    *   **`methodology.md`**: Details the specific formulas and steps used in the financial calculations (VaR/CVaR).
*   **`venv/`**: Directory containing the Python virtual environment (should be gitignored).
*   **`fixes/`** (Created if needed): Directory to store markdown files documenting solutions to complex or recurring bugs encountered during development.

## Basic Data Flow

1.  User interacts with `app.py` (Streamlit UI) - uploads CSV, sets parameters, clicks button.
2.  `app.py` calls functions in `risk_calculator.py` (or within `app.py` if not separated).
3.  `risk_calculator.py` functions:
    *   Load and parse CSV data.
    *   Fetch historical prices via `yfinance`.
    *   Perform return and risk calculations (using Pandas, NumPy, SciPy).
    *   Generate Plotly figure objects.
4.  Results (numeric values, Plotly figures) are returned to `app.py`.
5.  `app.py` uses Streamlit functions (`st.metric`, `st.plotly_chart`, etc.) to display the results to the user.