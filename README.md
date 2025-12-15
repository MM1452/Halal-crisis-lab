# Halal-crisis-lab
An interactive financial simulator built with Python & Streamlit. It stress-tests ethical portfolios against inflation, market crashes, and sector skew using historical data (2019-Present).
# 🛡️ Halal Crisis Lab | The Skeptic's Guide

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://halal-crisis-lab.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Halal investing is a quality filter, but it comes with a hidden cost."**
>
> This tool is the interactive companion to the Substack article: [The Wealth You Keep: A Skeptic's Guide](https://ardentmercator.substack.com/p/the-wealth-you-keep-a-skeptics-guide).

## 🧐 What is this?

Most investment charts show a smooth line going up and to the right. They ignore the panic of a crash, the erosion of inflation, and the drag of high fees.

The **Halal Crisis Lab** is a financial simulator designed to stress-test ethical (Shariah-compliant) portfolios against real-world market crises. It allows users to move beyond "vanity metrics" (Total Return) and analyze "pain metrics" (Max Drawdown, Real Return, and Sortino Ratio).

It answers the question: *Does excluding banks and high-debt companies actually protect your wealth during a crash?*

## 🚀 Key Features

### 1. The Scenario Engine
Instead of generic timeframes, users select specific historical crises to see how the portfolio reacts under pressure:
* **The Covid Crash (2020):** A stress test for volatility and tech resilience.
* **The Inflation Shock (2022):** What happens when interest rates rise and "Growth" stocks crash?
* **The AI Boom (2023):** Visualizing the impact of the "Magnificent 7."

### 2. The "Skeptic's" Controls
Standard calculators assume 0% inflation and 0% fees. This lab introduces **Friction**:
* **Inflation Slider:** Adjusts returns for purchasing power loss (0-15%).
* **"Cost of Conscience" Toggle:** Simulates the 0.55% annual drag of higher expense ratios + purification costs.

### 3. Advanced Risk Analytics
This tool moves beyond simple returns to educate users on professional risk metrics:
* **Sortino Ratio:** Measures downside risk (bad volatility) vs. total volatility.
* **Max Drawdown:** The "Pain Index"—how much value was lost from the peak.

### 4. The "Tech Proxy" Visualization
A visual overlay of the **Technology Sector (XLK)** against the Halal portfolio, proving the thesis that Shariah-compliant funds often act as a proxy for the Tech sector due to the exclusion of Financials.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Financial Data:** YFinance (Yahoo Finance API)
* **Visualization:** Plotly Express & Graph Objects
* **Language:** Python 3.10+

## 📊 Live Demo

**[Click here to launch the Crisis Lab](https://halal-crisis-lab.streamlit.app)**

*(Note: If you are seeing this on GitHub, the live app is hosted on Streamlit Community Cloud.)*
