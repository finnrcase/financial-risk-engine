"""
Global configuration settings for the Financial Risk Engine.

This file stores default parameters used across the project.
Keeping them here avoids hardcoding values in multiple modules.
"""

# Number of Monte Carlo simulations
DEFAULT_NUM_SIMULATIONS = 10000

# Trading days in a year
TRADING_DAYS_PER_YEAR = 252

# Default investment horizon
DEFAULT_TIME_HORIZON_DAYS = 252

# Default risk-free rate used for Sharpe ratio
DEFAULT_RISK_FREE_RATE = 0.02

# Default VaR / Expected Shortfall confidence levels
DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]

# Default initial portfolio value
DEFAULT_INITIAL_PORTFOLIO_VALUE = 1.0

# Plotting settings
DEFAULT_FIGURE_WIDTH = 10
DEFAULT_FIGURE_HEIGHT = 6