# Project COP: Financial Econometrics Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Status](https://img.shields.io/badge/status-Complete-brightgreen)

## 📊 Project Overview

Comprehensive econometric and financial analysis of energy sector enterprises, with primary focus on **ConocoPhillips (COP)**. This project addresses six interconnected research questions spanning valuation theory, time series forecasting, systematic risk assessment, and portfolio optimization.

**Course**: MF753 Financial Econometrics - Fall 2025  
**Institution**: Wilfrid Laurier University  
**Completion Date**: November 2025

---

## 🎯 Research Questions

| Question | Topic | Methodology | Key Result |
|----------|-------|-------------|-----------|
| **Q2** | System Risk Assessment | CAPM & Fama-French 3-Factor | COP β=1.247**, α=-3.74%/year |
| **Q3** | Valuation Metrics | Panel Regression + XGBoost | P/B R²=0.481 vs EV/EBITDA 0.181 (+166%) |
| **Q4** | Sales Forecasting (OLS) | OLS Regression | MAPE=16.24%, R²=0.704 |
| **Q4.2** | Sales Forecasting (ARIMA) | Box-Jenkins ARIMA(1,2,1) | MAPE=12.07%, AIC=919.67 |
| **Q5.2** | Return Distribution | Bootstrap Monte Carlo | P(Loss)=27.26%, Mean=31.91% |
| **Q5.4** | Portfolio Optimization | 4-Strategy Comparison | 50/50 Sharpe=0.9288 (optimal) |
| **Q6** | Volatility Dynamics | GARCH(1,1) Modeling | COP α+β=0.989 (high persistence) |

---

## 🔬 Key Methodologies

### Econometric Techniques
- **Panel Data**: Fixed-effects regression with company dummies (1,074 observations, 6 companies, 179 months)
- **Time Series**: Box-Jenkins ARIMA methodology with ADF/ACF/PACF diagnostics
- **Factor Models**: CAPM, Fama-French 3-Factor, COVID-19 structural break analysis
- **Diagnostics**: Durbin-Watson, Breusch-Pagan, Jarque-Bera, VIF, HAC correction

### Machine Learning
- **XGBoost**: Gradient boosting for non-linear valuation relationships
- **Feature Importance**: Ranking drivers of P/B and EV/EBITDA variations

### Simulation Methods
- **Bootstrap**: Non-parametric 10,000-trial Monte Carlo (no distribution assumptions)
- **GARCH(1,1)**: Parametric volatility clustering model
- **Portfolio Optimization**: 4 strategies × 10,000 simulations = 40,000 outcomes

---

## 📈 Key Findings

### 1. Valuation Framework (Q3)
**P/B outperforms EV/EBITDA decisively:**
- OLS: R² improvement of 166% (0.481 vs 0.181)
- XGBoost: R² improvement of 54% (0.836 vs 0.543)
- Reason: P/B stable during crises; EV/EBITDA denominator fails when EBITDA→0

### 2. Oil Price Dynamics (Q3)
**Markets respond to *momentum*, not *levels*:**
- Oil price MoM: β=+0.575*** (significant)
- Oil price level: β=-0.012 (not significant)
- Implication: Forward-looking market pricing captures momentum effects

### 3. Sales Forecasting (Q4/Q4.2)
**ARIMA superior for pure forecasting; OLS better for scenarios:**
- ARIMA MAPE: 12.07% (test set)
- OLS MAPE: 16.24% (test set)
- OLS advantage: Interpretable coefficients for "what-if" analysis
- ARIMA advantage: Captures univariate dynamics more efficiently

### 4. System Risk (Q2)
**COP exhibits high risk with below-market returns:**
- Market Beta: 1.247*** (24.7% above-market volatility)
- Jensen's Alpha: -3.74%/year (underperformance after risk adjustment)
- COVID Excess Loss: -1.847%/month (crisis vulnerability)
- Factor Profile: Large-cap growth characteristics (not value)

### 5. Portfolio Optimization (Q5)
**Diversification mathematically optimal:**
- Strategy C (50% COP + 50% S&P 500):
  - Sharpe Ratio: 0.9288 (highest)
  - Expected Return: 23.94% (captures 76% of 100% COP returns)
  - Risk: 24.69% (only 51% of 100% COP volatility)
  - Loss Probability: 16.23% (lowest among quality strategies)

### 6. Volatility Clustering (Q6)
**Energy stocks exhibit persistent volatility shocks:**
- COP GARCH α+β = 0.989 (high persistence)
- S&P 500 GARCH α+β = 0.896 (lower persistence)
- Implication: Shocks to energy volatility persist longer; important for risk forecasting

---

## 📁 Project Structure

```
Project-COP/
├── Project_Summary_EN.md           # Main technical report (comprehensive)
├── GITHUB_SETUP_GUIDE.md          # GitHub deployment guide
├── README.md                        # This file
│
├── Q2/                             # System Risk Assessment
│   ├── section2_summary.md
│   ├── section2_beta_xlwings.py
│   └── Workdone.xlsx
│
├── Q3/                             # Valuation Metrics
│   ├── README.md
│   ├── panel_regression_analysis.py
│   ├── Q3.csv
│   └── result/                     # 28 output files
│
├── Q4/                             # OLS Sales Forecasting
│   ├── README.md
│   ├── sales_forecast_final.py
│   ├── Q4.csv
│   └── sales_forecast_results.csv
│
├── Q4.2/                           # ARIMA Sales Forecasting
│   ├── ARMA_ANALYSIS_REPORT.md
│   ├── arma_forecast_final.py
│   └── arma_forecast_results.csv
│
├── Q5/                             # Portfolio Optimization
│   ├── q5_2_bootstrap_simulation.py
│   ├── q5_4_portfolio_strategies.py
│   ├── bootstrap_annual_returns.csv
│   ├── Q5_2_ANALYSIS_SUMMARY.txt
│   │
│   ├── Q5.4/                       # 4-Strategy Analysis
│   │   ├── COMPLETE_ANALYSIS.txt
│   │   ├── READ_ME_FIRST.txt
│   │   ├── Q5.4 Combined Data.xlsx (40,000 simulations)
│   │   ├── Risk_Return_Profile.png
│   │   └── [5 more visualization PNGs]
│   │
│   └── Q6_GARCH/                   # GARCH Volatility
│       ├── README_Q6_GARCH_CORRECTED.txt
│       ├── q6_garch_portfolio_strategies.py
│       ├── Q6_GARCH_Combined_Data.xlsx
│       └── [6 visualization PNGs]
```

---

## 🛠️ Technical Stack

- **Language**: Python 3.12
- **Data Processing**: pandas 2.0+, numpy
- **Econometrics**: statsmodels 0.14+
- **Machine Learning**: XGBoost
- **Time Series**: ARCH library (for GARCH)
- **Visualization**: matplotlib, seaborn
- **Simulation**: scipy, random (seed=42 for reproducibility)

---

## 📊 Data Sources

| Data | Source | Period | Frequency |
|------|--------|--------|-----------|
| Stock Returns (6 energy companies, SPX) | Yahoo Finance | 2010-2025 | Monthly (panel), Weekly (bootstrap) |
| Financial Data (EBITDA, Book Value, etc.) | Q3.csv | 2010-2025 | Quarterly |
| Sales Data | Q4.csv | 2012-2025 | Quarterly |
| WTI Crude Oil Prices | Standard market data | 2010-2025 | Monthly |
| Fama-French Factors | Kenneth French Data Library | 2020-2025 | Monthly |

---

## ⚠️ Important Disclaimer

**Academic Assignment**: This project is a final assignment for MF753 course at Wilfrid Laurier University.

**Not Investment Advice**: 
- This analysis is for educational purposes only
- NOT intended as investment advice or financial guidance
- Past performance does not predict future results
- All analysis based on specific historical sample period
- User assumes full responsibility for any decisions made based on this analysis

**Appropriate Use**:
- ✓ Academic research and learning
- ✓ Reference for econometric methodology
- ✗ Investment decision basis
- ✗ Financial advisory
- ✗ Commercial use without permission

---

## 🔍 Quality Assurance

**Econometric Standards**:
- ✓ Full diagnostic testing (Durbin-Watson, Breusch-Pagan, Jarque-Bera, VIF)
- ✓ HAC robust standard errors applied
- ✓ Out-of-sample validation performed
- ✓ Assumption checking documented

**Reproducibility**:
- ✓ Random seed fixed (seed=42)
- ✓ All code fully commented
- ✓ Data sources documented
- ✓ Results verified and validated

**Documentation**:
- ✓ Comprehensive technical report (Project_Summary_EN.md)
- ✓ Individual README files for each question
- ✓ Detailed result files and visualizations
- ✓ Clear methodology explanation

---

## 📖 How to Use This Repository

### Quick Start (5 minutes)
1. Read: `Project_Summary_EN.md` Executive Summary
2. View: Visualizations in `Q5.4/Risk_Return_Profile.png`
3. Review: Key findings in this README

### Technical Deep Dive (1-2 hours)
1. Read: Full `Project_Summary_EN.md`
2. Review: Individual Q*/README.md files
3. Examine: Source code (*.py files)
4. Analyze: Result data (.csv, .xlsx files)

### Reproducing Analysis
```bash
# Q2: Risk Assessment
python Q2/section2_beta_xlwings.py

# Q3: Valuation Analysis  
python Q3/panel_regression_analysis.py

# Q4: OLS Forecasting
python Q4/sales_forecast_final.py

# Q4.2: ARIMA Forecasting
python Q4.2/arma_forecast_final.py

# Q5: Portfolio Strategies
python Q5/q5_4_portfolio_strategies.py

# Q6: GARCH Volatility
python Q5/q6_garch_portfolio_strategies.py
```

---

## 📝 Academic References

**Econometric Theory**:
- Greene, W. H. (2012). *Econometric Analysis* (7th ed.)
- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis*
- Wooldridge, J. M. (2019). *Introductory Econometrics*

**Financial Theory**:
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds
- Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium
- Markowitz, H. (1952). Portfolio selection

**Applied Methods**:
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity
- Newey, W. K., & West, K. D. (1987). A simple positive semi-definite heteroskedasticity and autocorrelation consistent covariance matrix

---

## 👤 Author

**Student Project**: MF753 Financial Econometrics - Fall 2025  
**Institution**: Wilfrid Laurier University

---

## 📅 Project Timeline

- **Data Collection**: October 2025
- **Analysis**: October-November 2025
- **Documentation**: November 2025
- **Completion**: November 17, 2025
- **GitHub Upload**: November 2025

---

## 📞 Questions?

Refer to:
- `Project_Summary_EN.md` - Comprehensive technical documentation
- Individual `Q*/README.md` files - Question-specific details
- `GITHUB_SETUP_GUIDE.md` - Repository setup instructions

---

**Status**: ✅ Complete and Ready for Review

**Last Updated**: November 2025

---

*For educational purposes. Not financial advice. See disclaimer above.*

