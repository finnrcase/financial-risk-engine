# Financial Risk Engine

**Institutional Portfolio Risk Simulator**  
An interactive portfolio risk analytics platform built in Python and Streamlit for Monte Carlo simulation, stress testing, risk attribution, and mean-variance optimization.

## Live App

[https://your-streamlit-url-here.streamlit.app](https://financial-risk-engine-lrmaf4egwyhpoyrw7ivtmw.streamlit.app/)

---

## Overview

This Financial Risk Engine is a portfolio analytics tool designed to simulate portfolio performance under uncertainty and evaluate downside risk using industry-standard metrics sourced online.

The project uses a Monte Carlo simulation, measures tail risk with Value at Risk and Expected Shortfall, evaluates stress scenarios, decomposes asset-level risk contributions, and computes optimized portfolios.

## Features

### Core Risk Simulation
- Correlated Monte Carlo portfolio simulation
- Simulated portfolio path generation
- Terminal return distribution analysis
- Maximum drawdown estimation

### Risk Metrics
- Expected return
- Volatility
- VaR (95% and 99%)
- Expected Shortfall
- Probability of loss
- Maximum drawdown
- Sharpe ratio

### Stress Testing
Preset macro scenarios including:
- Baseline
- Market crash
- Rate spike
- Inflation shock
- Tech correction
- Energy shock

### Risk Attribution
- Marginal risk contribution
- Component risk contribution
- Percentage contribution to total portfolio risk

### Scenario Analysis
- Multi-scenario comparison table
- Risk metric comparison across scenarios

### Portfolio Optimization
- Minimum variance portfolio
- Maximum Sharpe portfolio
- Efficient frontier generation

### Interactive Dashboard
Built with Streamlit for:
- portfolio input controls
- simulation parameter selection
- scenario selection
- optimization analysis
- dynamic charts and tables

---

Example portfolio:

- **40% Equities**
- **30% Bonds**
- **20% Tech**
- **10% Crypto**

---

**Language**
- Python

**Core Libraries**
- NumPy
- pandas
- SciPy
- matplotlib
- Plotly
- Streamlit
- pytest

---

## Repository Structure

```text
financial-risk-engine/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── sample_asset_data.csv
│   ├── sample_correlation_matrix.csv
│   └── sample_portfolios.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── __init__.py
│   ├── portfolio.py
│   ├── simulation.py
│   ├── risk_metrics.py
│   ├── scenarios.py
│   ├── visualization.py
│   ├── validation.py
│   ├── utils.py
│   ├── risk_contribution.py
│   ├── scenario_analysis.py
│   └── optimization.py
├── tests/
│   ├── test_portfolio.py
│   ├── test_simulation.py
│   ├── test_risk_metrics.py
│   ├── test_scenarios.py
│   ├── test_risk_contribution.py
│   ├── test_scenario_analysis.py
│   └── test_optimization.py
├── config.py
├── conftest.py
├── requirements.txt
├── runtime.txt
└── README.md


