#!/bin/bash
# Quick Git Setup for Project COP
# Copy and paste these commands into PowerShell/Terminal

# ============================================================================
# STEP 1: Initialize Git Locally
# ============================================================================

git init
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"

# ============================================================================
# STEP 2: Add All Project Files
# ============================================================================

git add .

# ============================================================================
# STEP 3: Create Initial Commit
# ============================================================================

git commit -m "Initial commit: Project COP - Financial Econometrics Analysis

- Q2: CAPM & Fama-French Risk Assessment
- Q3: Valuation Metrics Panel Regression  
- Q4/Q4.2: Sales Forecasting (OLS & ARIMA)
- Q5: Portfolio Strategy Optimization (Bootstrap)
- Q6: GARCH Volatility Modeling

Data: December 2010 - October 2025
Course: MF753 Financial Econometrics Fall 2025"

# ============================================================================
# STEP 4: Connect to GitHub and Push
# ============================================================================

# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/Project-COP.git

git branch -M main

git push -u origin main

# ============================================================================
# Done! Your project is now on GitHub at:
# https://github.com/YOUR_USERNAME/Project-COP
# ============================================================================

# ============================================================================
# USEFUL COMMANDS FOR FUTURE UPDATES
# ============================================================================

# Check git status
# git status

# View commit history
# git log --oneline

# Make changes and push
# git add .
# git commit -m "Description of changes"
# git push

# Create a new branch for experiments
# git checkout -b feature-name
# git push -u origin feature-name

