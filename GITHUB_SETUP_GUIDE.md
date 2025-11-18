# GitHub Setup Guide - Project COP

## Step 1: Create GitHub Repository

### Online (at github.com):
1. Go to https://github.com/new
2. **Repository name**: `Project-COP` (or `project-cop`)
3. **Description**: `Financial Econometrics Analysis: Energy Sector Valuation, Forecasting & Portfolio Optimization (MF753 Course Project)`
4. **Visibility**: Choose `Public` (to share online) or `Private` (personal use)
5. **Initialize with**: Leave unchecked (we'll do it locally)
6. Click **Create repository**

### Copy the repository URL:
```
https://github.com/YOUR_USERNAME/Project-COP.git
```
(or SSH version if you have SSH key configured)

---

## Step 2: Prepare Your Local Files

Navigate to your project directory in PowerShell/Terminal:

```powershell
cd "C:\Users\Costa\OneDrive - Wilfrid Laurier University\Documents\Study material\753 financial econometrics\Assignment"
```

---

## Step 3: Initialize Git Repository Locally

```powershell
# Initialize git
git init

# Configure your Git identity (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

---

## Step 4: Create .gitignore File

Create a file named `.gitignore` to exclude unnecessary files:

```powershell
# Create .gitignore
Add-Content -Path ".gitignore" -Value @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data files (optional - keep if you want to include data)
*.csv
*.xlsx

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary
*.tmp
*~
"@
```

---

## Step 5: Add Files to Git

```powershell
# Add all relevant files
git add Project_Summary_EN.md
git add README.md (if you have one)
git add Q2/ Q3/ Q4/ Q4.2/ Q5/
git add *.py
git add *.txt

# Or add everything (respects .gitignore):
git add .

# Check what will be committed
git status
```

---

## Step 6: Create Initial Commit

```powershell
git commit -m "Initial commit: Project COP - Financial Econometrics Analysis

- Q2: CAPM & Fama-French Risk Assessment
- Q3: Valuation Metrics Panel Regression
- Q4/Q4.2: Sales Forecasting (OLS & ARIMA)
- Q5: Portfolio Strategy Optimization (Bootstrap)
- Q6: GARCH Volatility Modeling

Data: December 2010 - October 2025
Course: MF753 Financial Econometrics Fall 2025"
```

---

## Step 7: Connect to Remote Repository & Push

```powershell
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/Project-COP.git

# Rename branch to main (GitHub default)
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## Step 8: Create README.md for GitHub

Create a `README.md` file in your project root:

```powershell
Add-Content -Path "README.md" -Value @"
# Project COP: Financial Econometrics Analysis

Comprehensive econometric analysis of energy sector enterprises, focusing on ConocoPhillips (COP).

## Project Overview

This project addresses six interconnected research questions spanning valuation, forecasting, risk assessment, and portfolio optimization:

- **Q2**: System Risk Assessment (CAPM & Fama-French Models)
- **Q3**: Valuation Metrics Comparison (P/B vs EV/EBITDA)
- **Q4**: Sales Forecasting with OLS Regression
- **Q4.2**: Sales Forecasting with ARIMA Time Series
- **Q5**: Portfolio Strategy Optimization (Bootstrap Monte Carlo)
- **Q6**: GARCH Volatility Modeling

## Data Period

December 2010 - October 2025 (15 years)

## Key Findings

- P/B metric outperforms EV/EBITDA by 166% (R² improvement)
- ARIMA achieves 12.07% MAPE vs OLS 16.24% for quarterly forecasting
- 50/50 COP/S&P500 portfolio optimal (Sharpe ratio 0.9288)
- COP exhibits above-market systematic risk (β=1.247) with below-market returns (α=-3.74%/year)
- Strong volatility clustering in energy sector (α+β=0.989)

## Methodology

- Panel fixed-effects regression with XGBoost validation
- Box-Jenkins ARIMA time series modeling
- CAPM and Fama-French 3-factor models with HAC correction
- Non-parametric bootstrap and GARCH(1,1) simulations
- Out-of-sample validation and diagnostic testing

## Course

MF753 Financial Econometrics - Fall 2025
Wilfrid Laurier University

## Files

- `Project_Summary_EN.md` - Comprehensive technical analysis
- `Q2/` - Risk factor analysis
- `Q3/` - Valuation metrics
- `Q4/`, `Q4.2/` - Sales forecasting
- `Q5/` - Portfolio optimization
- Source code (Python) and result files

## Important Note

This is an academic course project for educational purposes. It is not investment advice and should not be used for actual investment decisions.

---

**Analysis Date**: November 2025  
**Last Updated**: November 2025
"@
```

Then add and commit:

```powershell
git add README.md
git commit -m "Add project README"
git push
```

---

## Step 9: Verify on GitHub

Visit: `https://github.com/YOUR_USERNAME/Project-COP`

You should see:
- README.md displayed
- Project files listed
- Full repository history

---

## Optional: Add Project Description on GitHub

1. Go to your repository settings
2. Under "About", add:
   - **Description**: Financial Econometrics Analysis - Energy Sector (MF753)
   - **Website**: (leave blank or add personal website)
   - **Topics**: Add `econometrics`, `finance`, `energy`, `forecasting`, `portfolio-optimization`

---

## Useful Git Commands for Future Updates

```powershell
# Check status
git status

# View recent commits
git log --oneline -5

# Make changes and push
git add .
git commit -m "Description of changes"
git push

# Create a new branch for experiments
git checkout -b feature-branch
git add .
git commit -m "New feature"
git push -u origin feature-branch
```

---

## Quick Reference: Complete Setup Script

```powershell
# Navigate to project folder
cd "Your\Project\Path"

# Initialize
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Create .gitignore and README
# (use Add-Content commands above)

# Add and commit
git add .
git commit -m "Initial commit: Project COP"

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/Project-COP.git
git branch -M main
git push -u origin main
```

Done! Your project is now on GitHub! 🚀


