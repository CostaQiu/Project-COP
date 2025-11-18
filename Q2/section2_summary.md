# Section 2: Company Beta Estimation - Complete Guide

## Overview

This script performs comprehensive beta estimation using CAPM and Fama-French 3-Factor models with diagnostic tests, HAC correction, and visual residual analysis.

**Script**: `section2_beta_xlwings.py`  
**Data**: 60 monthly observations from `Workdone.xlsx` (Sheet "2")  
**Output**: `Workdone.xlsx` → Sheet "2-result"

---

## Models Implemented

### 1. **CAPM Model**
- **Formula**: `R_i = α + β_M × (Mkt-RF) + ε`
- **Purpose**: Estimates market beta (systematic risk)
- **Variables**: All companies (COP, XOM, CVX, EOG, BP, DVN, ^cl, ^HG, ^SP500-10, ^SPX, Peers Average)

### 2. **Fama-French 3-Factor Model (FF3)**
- **Formula**: `R_i = α + β_M × (Mkt-RF) + β_S × SMB + β_H × HML + ε`
- **Purpose**: Captures market, size, and value risk factors
- **Variables**: All companies

### 3. **Fama-French 3-Factor with COVID Dummy (FF3_COVID)**
- **Formula**: `R_i = α + β_M × (Mkt-RF) + β_S × SMB + β_H × HML + β_COVID × Covid + ε`
- **Purpose**: Tests for average return difference during COVID period
- **Note**: COVID is a dummy variable (0/1), not an interaction term
- **Output includes**:
  - `beta_Mkt`: Market beta (constant across periods)
  - `beta_SMB`: Size factor loading
  - `beta_HML`: Value factor loading
  - `beta_Covid`: Average return differential during COVID periods

---

## Diagnostic Tests and HAC Correction

### Tests Performed

#### 1. **Durbin-Watson Test** (Autocorrelation)
- **Purpose**: Detects serial correlation in residuals
- **Ideal Value**: ≈ 2.0
- **Interpretation**:
  - DW < 1.5: Positive autocorrelation
  - DW > 2.5: Negative autocorrelation
  - 1.5 ≤ DW ≤ 2.5: No significant autocorrelation

#### 2. **Breusch-Pagan Test** (Heteroskedasticity)
- **Purpose**: Tests whether error variance depends on independent variables
- **Interpretation**:
  - p-value < 0.05: Heteroskedasticity detected
  - p-value ≥ 0.05: Homoskedasticity (constant variance)

### HAC Correction Trigger

**HAC (Heteroskedasticity and Autocorrelation Consistent) robust standard errors are applied when:**

- **Durbin-Watson < 1.5 OR Durbin-Watson > 2.5** (autocorrelation detected), **OR**
- **Breusch-Pagan p-value < 0.05** (heteroskedasticity detected)

### What Changes with HAC Correction

| Element | Changes? | Notes |
|---------|----------|-------|
| **Coefficients (β)** | ❌ No | Point estimates remain identical |
| **Standard Errors** | ✅ Yes | Adjusted to be robust to heteroskedasticity & autocorrelation |
| **t-statistics** | ✅ Yes | Recalculated as: t = β / robust_SE |
| **p-values** | ✅ Yes | Updated based on new t-statistics (see `pval_Mkt_corrected`) |
| **Confidence Intervals** | ✅ Yes | Recalculated using robust standard errors |
| **R-squared** | ❌ No | Model fit unchanged |
| **F-statistic** | ❌ No | Overall model significance unchanged |

**Key Point**: HAC correction affects **inference** (hypothesis testing), not the estimated relationships (coefficients).

### Output Structure in Excel (`2-result` sheet)

#### Part 1: Beta Summary Tables (All Variables)
Three summary tables showing betas for all companies:
1. **CAPM - Beta Summary**: Shows α and β_M for each company
2. **FF3 - Beta Summary**: Shows α, β_M, β_S, β_H for each company
3. **FF3_COVID - Beta Summary**: Shows α, β_Mkt_pre, β_Mkt_diff, β_Mkt_covid, β_SMB, β_HML

#### Part 2: Comprehensive Regression Output (Selected Variables Only)
For **COP**, **^SP500-10**, and **Peers Average** only, each model includes:

1. **Model Summary Statistics**:
   - R-squared
   - Adjusted R-squared
   - F-statistic
   - Prob (F-statistic)
   - AIC (Akaike Information Criterion)
   - BIC (Bayesian Information Criterion)
   - N (Number of observations)

2. **Coefficients Table**:
   - Variable names
   - Coefficient estimates
   - Standard errors
   - t-statistics
   - P-values
   - 95% Confidence intervals [0.025, 0.975]

3. **ANOVA Table**:
   - Regression/Residual/Total sum of squares
   - Degrees of freedom
   - Mean squares
   - F-statistic
   - P-value

---

## Output Format

### Numerical Precision
- **All values displayed with 3 decimal places** (e.g., 0.123, 1.234, 0.001)
- Provides sufficient precision for beta coefficients, p-values, and test statistics

### Visual Organization
- **Color-coded by variable** for easy navigation:
  - **COP**: Light blue background
  - **^SP500-10**: Light green background
  - **Peers Average**: Light orange background
- **HAC-corrected tables**: Slightly darker shade for distinction
- **Residual plots**: Positioned to the right of regression output

---

## Key Features

1. ✅ **All 60 observations used** (COVID zeros treated as valid non-COVID periods)
2. ✅ **3 decimal places** for all numerical output
3. ✅ **Comprehensive diagnostic tests** (Durbin-Watson, Breusch-Pagan)
4. ✅ **Automatic HAC correction** when violations detected
5. ✅ **Residual diagnostic plots** for visual inspection
6. ✅ **Both original and corrected results** when applicable
7. ✅ **Complete ANOVA tables** for model assessment
8. ✅ **Beta summary for all variables** with p-values (original and corrected)

---

## How to Run

### Prerequisites
```bash
pip install xlwings pandas statsmodels matplotlib pillow
```

### Execution
```bash
python section2_beta_xlwings.py
```

### What the Script Does
1. Opens `Workdone.xlsx`
2. Reads 60 monthly observations from sheet "2" (columns L through AB)
3. Runs 3 models (CAPM, FF3, FF3_COVID) for all 11 variables
4. Performs diagnostic tests (Durbin-Watson, Breusch-Pagan)
5. Applies HAC correction automatically when needed
6. Generates residual plots for 3 target variables
7. Writes all results to sheet "2-result"
8. Saves automatically

### Output Location
Open `Workdone.xlsx` → Navigate to sheet **"2-result"**

---

## Interpreting Results

### Beta Summary Tables
- **alpha**: Intercept (excess return not explained by factors)
- **beta_Mkt**: Market beta (systematic risk, β > 1 means more volatile than market)
- **beta_SMB**: Size factor loading (positive = small-cap tilt)
- **beta_HML**: Value factor loading (positive = value stock tilt)
- **beta_Covid**: Average return differential during COVID (positive = outperformed during COVID)
- **pval_Mkt**: Original p-value for market beta
- **pval_Mkt_corrected**: HAC-corrected p-value (use this if correction was applied)
- **R2_adj**: Adjusted R-squared (model fit, higher is better)

### Diagnostic Tests
- **Durbin-Watson**: Check if close to 2.0 (no autocorrelation)
- **Breusch-Pagan p-value**: Check if > 0.05 (no heteroskedasticity)
- **Correction Applied**: "Yes (HAC)" means robust standard errors were used

### When to Use Corrected Results
- If "Correction Applied" = "Yes (HAC)", use the **HAC Corrected** coefficient table
- Coefficients (β) are the same, but standard errors, t-stats, and p-values are more reliable
- Use `pval_Mkt_corrected` from the summary table for hypothesis testing

### Residual Plots
- **Plot 1 (Time)**: Look for patterns over time (should be random)
- **Plot 2 (Mkt-RF)**: Look for fan shapes or patterns (should be random scatter)
- Patterns indicate model violations (autocorrelation or heteroskedasticity)

---

## Technical Notes

- **Data handling**: COVID column NaN values treated as 0 (non-COVID periods)
- **Missing values**: Listwise deletion (complete cases only)
- **HAC method**: Newey-West with maxlags=1
- **Degrees of freedom**: 
  - CAPM: df = 60 - 2 = 58
  - FF3: df = 60 - 4 = 56
  - FF3_COVID: df = 60 - 5 = 55

---

## Contact & Support

For questions about the methodology or results, refer to:
- **Statsmodels documentation**: https://www.statsmodels.org/
- **Fama-French factors**: Kenneth French Data Library
- **HAC standard errors**: Newey & West (1987)

