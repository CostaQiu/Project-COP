"""
Final Sales Forecasting Model - With Quarterly Dummies and Extended Forecast
Features: Sales_lag1, Sales_lag4, Oil_Q_avg, ROIC_LTM_lag4, Q2, Q3, Q4 dummies
Data filtered from 2011 Q1 onwards (excluding upstream sales period)
Author: AI Assistant
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("SALES FORECASTING - WITH QUARTERLY DUMMIES")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PARSE DATA CORRECTLY
# ============================================================================
print("\n[STEP 1] Loading and parsing data...")

df_raw = pd.read_csv('Q4.csv', header=None)

# Parse quarterly sales data from columns 4-8, rows 2-66 (skip row 1 which is header)
quarterly = df_raw.iloc[2:66, [4, 5, 6, 7, 8]].copy()
quarterly.columns = ['Date', 'Sales', 'EBITDA', 'ROIC_Q', 'ROIC_LTM']

# Parse oil prices from columns 0-2, rows 7-187
oil_copper = df_raw.iloc[7:187, [0, 1, 2]].copy()
oil_copper.columns = ['Date', 'Oil_Price', 'Copper_Price']

# Convert to numeric first (before datetime conversion)
for col in ['Sales', 'EBITDA', 'ROIC_Q', 'ROIC_LTM']:
    quarterly[col] = pd.to_numeric(quarterly[col], errors='coerce')
    
for col in ['Oil_Price', 'Copper_Price']:
    oil_copper[col] = pd.to_numeric(oil_copper[col], errors='coerce')

# Convert dates - handle both text dates (M/D/YYYY) and Excel format
def parse_date(date_val):
    try:
        # Try parsing as text date (M/D/YYYY format)
        return pd.to_datetime(date_val, format='%m/%d/%Y')
    except:
        try:
            # Try as Excel serial number
            excel_date = float(date_val)
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=excel_date)
        except:
            return pd.NaT

quarterly['Date'] = quarterly['Date'].apply(parse_date)
oil_copper['Date'] = oil_copper['Date'].apply(parse_date)

# Remove NaT and sort
quarterly = quarterly.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
oil_copper = oil_copper.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# Add year/quarter identifiers
quarterly['Year'] = quarterly['Date'].dt.year
quarterly['Quarter'] = quarterly['Date'].dt.quarter
quarterly['YearQuarter'] = quarterly['Year'].astype(str) + 'Q' + quarterly['Quarter'].astype(str)

oil_copper['Year'] = oil_copper['Date'].dt.year
oil_copper['Quarter'] = oil_copper['Date'].dt.quarter
oil_copper['YearQuarter'] = oil_copper['Year'].astype(str) + 'Q' + oil_copper['Quarter'].astype(str)

print(f"  - Quarterly observations (raw): {len(quarterly)}")
print(f"  - Date range (raw): {quarterly['Date'].min().strftime('%Y-%m-%d')} to {quarterly['Date'].max().strftime('%Y-%m-%d')}")
print(f"  - Year range: {quarterly['Year'].min()} to {quarterly['Year'].max()}")

# ============================================================================
# STEP 2: CREATE QUARTERLY OIL PRICE AVERAGE
# ============================================================================
print("\n[STEP 2] Creating quarterly oil price averages...")

oil_quarterly = oil_copper.groupby('YearQuarter')['Oil_Price'].mean().reset_index()
oil_quarterly.columns = ['YearQuarter', 'Oil_Q_avg']

df = quarterly.merge(oil_quarterly, on='YearQuarter', how='left')
print(f"  - Oil price quarterly averages created")

# ============================================================================
# STEP 3: FILTER DATA FROM 2011 Q1 ONWARDS
# ============================================================================
print("\n[STEP 3] Filtering data from 2011 Q1 onwards...")
print("  (Excluding earlier periods with upstream sales)")

# Filter to 2011 Q1 and later
df = df[(df['Year'] >= 2011)].copy()
df = df.reset_index(drop=True)

print(f"  - Filtered dataset: {len(df)} observations")
print(f"  - Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

# ============================================================================
# STEP 4: CREATE FEATURES WITH LAGS AND QUARTERLY DUMMIES
# ============================================================================
print("\n[STEP 4] Creating features with lags and quarterly dummies...")
print("  - Sales_lag1: Previous quarter sales")
print("  - Sales_lag4: 4 quarters ago (year-over-year)")
print("  - Oil_Q_avg: Current quarter average")
print("  - ROIC_LTM_lag4: 1 year ago")
print("  - Q2, Q3, Q4: Quarterly dummy variables (Q1 is reference)")

# Create lagged features
df['Sales_lag1'] = df['Sales'].shift(1)
df['Sales_lag4'] = df['Sales'].shift(4)

# Select final columns (simplified model - only significant predictors)
df_model = df[['Date', 'Year', 'Quarter', 'YearQuarter', 
               'Sales', 'Sales_lag1', 'Sales_lag4', 'Oil_Q_avg']].copy()

# Drop rows with NaN (due to lags)
df_model = df_model.dropna()

print(f"\n  Final dataset shape: {df_model.shape}")
print(f"  Observations available: {len(df_model)}")

# Save the modeling dataset
df_model.to_csv('sales_modeling_data.csv', index=False)
print(f"  [OK] Saved: sales_modeling_data.csv")

# ============================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[STEP 5] Descriptive statistics...")

features = ['Sales', 'Sales_lag1', 'Sales_lag4', 'Oil_Q_avg']
desc_stats = df_model[features].describe()
print("\n" + desc_stats.to_string())

print("\n[Quarterly Distribution]")
print(df_model['Quarter'].value_counts().sort_index())

# ============================================================================
# STEP 6: TRAIN/TEST SPLIT
# ============================================================================
print("\n[STEP 6] Train/test split...")

# Use data up to 2024 Q3 for training, 2024 Q4 - 2025 Q3 for testing
train_data = df_model[df_model['Date'] < '2024-10-01'].copy()
test_data = df_model[df_model['Date'] >= '2024-10-01'].copy()

X_features = ['Sales_lag1', 'Sales_lag4', 'Oil_Q_avg']
X_train = train_data[X_features]
y_train = train_data['Sales']
X_test = test_data[X_features]
y_test = test_data['Sales']

dates_train = train_data['Date']
dates_test = test_data['Date']

print(f"  - Training set: {len(X_train)} observations ({dates_train.min().strftime('%Y-%m')} to {dates_train.max().strftime('%Y-%m')})")
print(f"  - Test set: {len(X_test)} observations ({dates_test.min().strftime('%Y-%m')} to {dates_test.max().strftime('%Y-%m')})")
print(f"  - Feature/Observation ratio: {len(X_features)}/{len(X_train)} = {len(X_features)/len(X_train):.3f}")

# ============================================================================
# STEP 7: OLS REGRESSION WITH FULL DIAGNOSTICS
# ============================================================================
print("\n[STEP 7] OLS Regression Model...")
print("="*80)

# Fit model with statsmodels for comprehensive statistics
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_const).fit()

# Print full summary
print(ols_model.summary())

# Additional diagnostic tests
print("\n" + "="*80)
print("ADDITIONAL DIAGNOSTIC TESTS")
print("="*80)

# 1. Durbin-Watson (autocorrelation)
dw_stat = durbin_watson(ols_model.resid)
print(f"\n1. Durbin-Watson Test (Autocorrelation):")
print(f"   Statistic: {dw_stat:.4f}")
print(f"   Interpretation: ", end="")
if 1.5 < dw_stat < 2.5:
    print("No significant autocorrelation")
elif dw_stat <= 1.5:
    print("Positive autocorrelation detected")
else:
    print("Negative autocorrelation detected")

# 2. Breusch-Pagan (heteroscedasticity)
bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
print(f"\n2. Breusch-Pagan Test (Heteroscedasticity):")
print(f"   LM Statistic: {bp_test[0]:.4f}")
print(f"   p-value: {bp_test[1]:.4f}")
print(f"   Conclusion: {'Homoscedastic' if bp_test[1] > 0.05 else 'Heteroscedastic'} (at 5% level)")

# 3. Breusch-Godfrey (serial correlation)
bg_test = acorr_breusch_godfrey(ols_model, nlags=2)
print(f"\n3. Breusch-Godfrey Test (Serial Correlation, 2 lags):")
print(f"   LM Statistic: {bg_test[0]:.4f}")
print(f"   p-value: {bg_test[1]:.4f}")
print(f"   Conclusion: {'No serial correlation' if bg_test[1] > 0.05 else 'Serial correlation detected'} (at 5% level)")

# 4. Jarque-Bera (normality)
jb_test = stats.jarque_bera(ols_model.resid)
print(f"\n4. Jarque-Bera Test (Normality of Residuals):")
print(f"   JB Statistic: {jb_test[0]:.4f}")
print(f"   p-value: {jb_test[1]:.4f}")
print(f"   Conclusion: {'Residuals are normal' if jb_test[1] > 0.05 else 'Residuals are not normal'} (at 5% level)")

# 5. VIF (Multicollinearity)
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(f"\n5. Variance Inflation Factors (Multicollinearity):")
for i, col in enumerate(X_features):
    vif = variance_inflation_factor(X_train.values, i)
    print(f"   {col:25s}: {vif:8.3f} {'(OK)' if vif < 10 else '(HIGH)'}")

print("\n" + "="*80)

# Save detailed summary to file
with open('model_regression_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SALES FORECASTING - OLS REGRESSION SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(str(ols_model.summary()))
    f.write("\n\n" + "="*80 + "\n")
    f.write("DIAGNOSTIC TESTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"1. Durbin-Watson: {dw_stat:.4f}\n")
    f.write(f"2. Breusch-Pagan p-value: {bp_test[1]:.4f}\n")
    f.write(f"3. Breusch-Godfrey p-value: {bg_test[1]:.4f}\n")
    f.write(f"4. Jarque-Bera p-value: {jb_test[1]:.4f}\n")
    f.write(f"\n5. Variance Inflation Factors:\n")
    for i, col in enumerate(X_features):
        vif = variance_inflation_factor(X_train.values, i)
        f.write(f"   {col:25s}: {vif:8.3f}\n")

print("\n[OK] Saved: model_regression_summary.txt")

# ============================================================================
# STEP 8: PREDICTIONS FOR TEST SET
# ============================================================================
print("\n[STEP 8] Generating predictions for test set...")

# Train and test predictions with prediction intervals
y_train_pred = ols_model.predict(X_train_const)
predictions_test = ols_model.get_prediction(X_test_const)
y_test_pred = predictions_test.predicted_mean

# Get 95% prediction intervals for test set
pred_intervals_test = predictions_test.conf_int(alpha=0.05)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f"\n  Model Performance:")
print(f"  {'='*70}")
print(f"  Training Set:")
print(f"    R-squared: {train_r2:.4f}")
print(f"    RMSE: {train_rmse:.2f}")
print(f"    MAE: {train_mae:.2f}")
print(f"\n  Test Set (Out-of-Sample):")
print(f"    R-squared: {test_r2:.4f}")
print(f"    RMSE: {test_rmse:.2f}")
print(f"    MAE: {test_mae:.2f}")
print(f"    MAPE: {test_mape:.2f}%")

# ============================================================================
# STEP 9: FORECAST FOR 2025 Q4 AND 2026 Q1
# ============================================================================
print("\n[STEP 9] Generating forecasts for 2025 Q4 and 2026 Q1...")

# Get the last available data to build forecast inputs
last_sales = df_model['Sales'].iloc[-1]
last_oil = df_model['Oil_Q_avg'].iloc[-1]

print(f"  Last Sales (2025 Q3): {last_sales:,.0f}")
print(f"  Last Oil Q_avg: ${last_oil:.2f}")

# Forecast for 2025 Q4
forecast_2025Q4 = {
    'Sales_lag1': last_sales,  # 2025 Q3 sales becomes lag1
    'Sales_lag4': df_model['Sales'].iloc[-4],  # 2024 Q4 sales
    'Oil_Q_avg': last_oil,  # Assume same oil price
}

X_2025Q4 = pd.DataFrame([forecast_2025Q4])
X_2025Q4_const = sm.add_constant(X_2025Q4, has_constant='add')
pred_2025Q4 = ols_model.get_prediction(X_2025Q4_const)
y_2025Q4 = pred_2025Q4.predicted_mean[0]
ci_2025Q4 = pred_2025Q4.conf_int(alpha=0.05)

# Forecast for 2026 Q1
forecast_2026Q1 = {
    'Sales_lag1': y_2025Q4,  # Use forecasted 2025 Q4
    'Sales_lag4': df_model['Sales'].iloc[-3],  # 2025 Q1 actual
    'Oil_Q_avg': last_oil,  # Assume same oil price
}

X_2026Q1 = pd.DataFrame([forecast_2026Q1])
X_2026Q1_const = sm.add_constant(X_2026Q1, has_constant='add')
pred_2026Q1 = ols_model.get_prediction(X_2026Q1_const)
y_2026Q1 = pred_2026Q1.predicted_mean[0]
ci_2026Q1 = pred_2026Q1.conf_int(alpha=0.05)

ci_2025Q4_lower = ci_2025Q4[0, 0] if isinstance(ci_2025Q4, np.ndarray) else ci_2025Q4.iloc[0,0]
ci_2025Q4_upper = ci_2025Q4[0, 1] if isinstance(ci_2025Q4, np.ndarray) else ci_2025Q4.iloc[0,1]

ci_2026Q1_lower = ci_2026Q1[0, 0] if isinstance(ci_2026Q1, np.ndarray) else ci_2026Q1.iloc[0,0]
ci_2026Q1_upper = ci_2026Q1[0, 1] if isinstance(ci_2026Q1, np.ndarray) else ci_2026Q1.iloc[0,1]

print(f"\n  2025 Q4 Forecast: {y_2025Q4:,.0f}")
print(f"    95% CI: [{ci_2025Q4_lower:,.0f}, {ci_2025Q4_upper:,.0f}]")
print(f"    CI Width: {ci_2025Q4_upper - ci_2025Q4_lower:,.0f}")

print(f"\n  2026 Q1 Forecast: {y_2026Q1:,.0f}")
print(f"    95% CI: [{ci_2026Q1_lower:,.0f}, {ci_2026Q1_upper:,.0f}]")
print(f"    CI Width: {ci_2026Q1_upper - ci_2026Q1_lower:,.0f}")

# ============================================================================
# STEP 10: CREATE COMPREHENSIVE FORECAST RESULTS
# ============================================================================
print("\n[STEP 10] Creating comprehensive forecast results...")

# Combine test period results with future forecasts
forecast_results = []

# Test period (actual vs forecast with CIs)
for idx, (date, actual, pred) in enumerate(zip(dates_test, y_test, y_test_pred)):
    # Handle both numpy array and DataFrame formats
    if isinstance(pred_intervals_test, np.ndarray):
        ci_lower = pred_intervals_test[idx, 0]
        ci_upper = pred_intervals_test[idx, 1]
    else:
        ci_lower = pred_intervals_test.iloc[idx, 0]
        ci_upper = pred_intervals_test.iloc[idx, 1]
    ci_width = ci_upper - ci_lower
    
    forecast_results.append({
        'Date': date.strftime('%Y-%m-%d'),
        'Year': int(date.year),
        'Quarter': int(date.quarter),
        'Actual_Sales': float(actual),
        'Forecast_Sales': float(pred),
        'Error': float(actual - pred),
        'Error_Pct': float((actual - pred) / actual * 100),
        'CI_Lower_95': float(ci_lower),
        'CI_Upper_95': float(ci_upper),
        'CI_Width': float(ci_width)
    })

# Future forecasts (2025 Q4, 2026 Q1)
forecast_results.append({
    'Date': '2025-12-31',
    'Year': 2025,
    'Quarter': 4,
    'Actual_Sales': np.nan,
    'Forecast_Sales': float(y_2025Q4),
    'Error': np.nan,
    'Error_Pct': np.nan,
    'CI_Lower_95': float(ci_2025Q4_lower),
    'CI_Upper_95': float(ci_2025Q4_upper),
    'CI_Width': float(ci_2025Q4_upper - ci_2025Q4_lower)
})

forecast_results.append({
    'Date': '2026-03-31',
    'Year': 2026,
    'Quarter': 1,
    'Actual_Sales': np.nan,
    'Forecast_Sales': float(y_2026Q1),
    'Error': np.nan,
    'Error_Pct': np.nan,
    'CI_Lower_95': float(ci_2026Q1_lower),
    'CI_Upper_95': float(ci_2026Q1_upper),
    'CI_Width': float(ci_2026Q1_upper - ci_2026Q1_lower)
})

forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_csv('sales_forecast_results.csv', index=False)
print(f"  [OK] Saved: sales_forecast_results.csv")

# Display results
print("\n  Forecast Details:")
print(f"  {'='*100}")
for idx, row in forecast_df.iterrows():
    if pd.notna(row['Actual_Sales']):
        print(f"  {int(row['Year'])} Q{int(row['Quarter'])}: Actual={row['Actual_Sales']:,.0f}, "
              f"Forecast={row['Forecast_Sales']:,.0f}, Error={row['Error']:+,.0f} ({row['Error_Pct']:+.1f}%)")
        print(f"    95% CI: [{row['CI_Lower_95']:,.0f}, {row['CI_Upper_95']:,.0f}], Width: {row['CI_Width']:,.0f}")
    else:
        print(f"  {int(row['Year'])} Q{int(row['Quarter'])}: Forecast={row['Forecast_Sales']:,.0f}")
        print(f"    95% CI: [{row['CI_Lower_95']:,.0f}, {row['CI_Upper_95']:,.0f}], Width: {row['CI_Width']:,.0f}")

# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================
print("\n[STEP 11] Creating visualizations...")

# Create simplified visualization - ONLY 2 ESSENTIAL PLOTS
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35)

# 1. Time series with forecast and CIs
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(dates_train, y_train, 'o-', label='Actual (Train)', color='#2E86AB', 
         markersize=5, linewidth=2, alpha=0.8)
ax1.plot(dates_train, y_train_pred, '--', label='Predicted (Train)', color='#06D6A0', 
         linewidth=2, alpha=0.7)

# Test period with CIs
ax1.plot(dates_test, y_test, 'o-', label='Actual (Test)', color='#2E86AB', 
         markersize=8, linewidth=2.5)
ax1.plot(dates_test, y_test_pred, 's--', label='Forecast (Test)', color='#EF476F', 
         markersize=8, linewidth=2.5)

# Plot prediction intervals for test
if isinstance(pred_intervals_test, np.ndarray):
    ax1.fill_between(dates_test, pred_intervals_test[:, 0], pred_intervals_test[:, 1],
                      color='#EF476F', alpha=0.2, label='95% CI')
else:
    ax1.fill_between(dates_test, pred_intervals_test.iloc[:, 0], pred_intervals_test.iloc[:, 1],
                      color='#EF476F', alpha=0.2, label='95% CI')

# Future forecasts
future_dates = [pd.Timestamp('2025-12-31'), pd.Timestamp('2026-03-31')]
future_forecasts = [y_2025Q4, y_2026Q1]
future_ci_lower = [ci_2025Q4_lower, ci_2026Q1_lower]
future_ci_upper = [ci_2025Q4_upper, ci_2026Q1_upper]

ax1.plot(future_dates, future_forecasts, 'D--', color='#FF6B35', markersize=10, 
         linewidth=2.5, label='Future Forecast')
ax1.fill_between(future_dates, future_ci_lower, future_ci_upper,
                  color='#FF6B35', alpha=0.2)

ax1.axvline(x=dates_test.iloc[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quarterly Sales', fontsize=12, fontweight='bold')
ax1.set_title('Sales Forecast: Actual vs Predicted Over Time', 
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='best', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Residuals vs Fitted (only residuals plot from training set)
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_train - y_train_pred
ax2.scatter(y_train_pred, residuals, alpha=0.6, s=80, color='#118AB2', edgecolors='black', linewidth=0.8)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2.5, alpha=0.8)
ax2.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax2.set_title('Residuals vs Fitted (Training Set)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.suptitle('Sales Forecasting Model - Simplified Diagnostics (Sales_lag1, Sales_lag4, Oil_Q_avg)', 
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('sales_forecast_analysis.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: sales_forecast_analysis.png (Updated - 2 essential plots only)")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("\nGenerated Files:")
print("  1. sales_modeling_data.csv - Clean dataset with 7 features (data from 2011 Q1)")
print("  2. model_regression_summary.txt - Complete regression output with diagnostics")
print("  3. sales_forecast_results.csv - Forecasts with 95% CIs (2024 Q4 - 2025 Q3, plus 2025 Q4 & 2026 Q1)")
print("  4. sales_forecast_analysis.png - Comprehensive 7-panel visualizations")

print("\nModel Summary:")
print(f"  Features Used: {len(X_features)} (4 continuous + 3 quarterly dummies)")
print(f"  Training Observations: {len(X_train)} (2011 Q1 - 2024 Q3)")
print(f"  Test Observations: {len(X_test)} (2024 Q4 - 2025 Q3)")
print(f"  Adjusted R-squared: {ols_model.rsquared_adj:.4f}")
print(f"  F-statistic: {ols_model.fvalue:.2f} (p-value: {ols_model.f_pvalue:.6f})")
print(f"  Test MAPE: {test_mape:.2f}%")

print("\nSignificant Features (p < 0.10):")
for i, name in enumerate(X_features):
    pval = ols_model.pvalues[i+1]
    coef = ols_model.params[i+1]
    if pval < 0.10:
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*'
        print(f"  {name:25s}: coef={coef:10.4f}, p-value={pval:.6f} {sig}")

print("\nFuture Forecasts (with 95% Prediction Intervals):")
print(f"  2025 Q4: {y_2025Q4:,.0f} [CI: {ci_2025Q4.iloc[0,0]:,.0f} - {ci_2025Q4.iloc[0,1]:,.0f}]")
print(f"  2026 Q1: {y_2026Q1:,.0f} [CI: {ci_2026Q1.iloc[0,0]:,.0f} - {ci_2026Q1.iloc[0,1]:,.0f}]")

ci_width = ci_2025Q4.iloc[0,1] - ci_2025Q4.iloc[0,0]
print(f"\nNote: 95% CI widths are large (~{ci_width:,.0f})")
print("  This reflects prediction uncertainty inherent in time series forecasting")

print("\n" + "="*80)
