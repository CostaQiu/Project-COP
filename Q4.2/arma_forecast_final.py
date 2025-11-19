"""
ARMA Time Series Forecasting - Box-Jenkins Methodology
Quarterly Sales Forecasting (2012 Q3 - 2026 Q1)

Key improvements:
1. Data starts from 2012 Q3 to avoid insufficient lag issues
2. ACF/PACF diagnostic plots for model identification
3. Multiple differencing levels tested (d=0,1,2)
4. Comprehensive model diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Time series analysis
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*80)
print("ARMA FORECASTING - BOX-JENKINS METHODOLOGY")
print("Data Period: 2012 Q3 - 2026 Q1")
print("="*80)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[Step 1] Loading and parsing data...")

def parse_date(date_val):
    """Parse Excel serial dates or text dates"""
    try:
        return pd.to_datetime(date_val, format='%m/%d/%Y')
    except:
        try:
            excel_date = float(date_val)
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=excel_date)
        except:
            return pd.NaT

# Read CSV
df_raw = pd.read_csv('../Q4/Q4.csv', header=None)
quarterly_data = df_raw.iloc[2:62, 4:9].copy()  # Skip header row
quarterly_data.columns = ['Date', 'Sales', 'EBITDA', 'ROIC_Q', 'ROIC_LTM']
quarterly_data['Date'] = quarterly_data['Date'].apply(parse_date)
quarterly_data = quarterly_data.dropna(subset=['Date']).reset_index(drop=True)
quarterly_data = quarterly_data.sort_values('Date').reset_index(drop=True)

# Filter from 2012 Q3 (to avoid insufficient lag data causing outliers)
cutoff_date = pd.Timestamp('2012-07-01')
df_series = quarterly_data[quarterly_data['Date'] >= cutoff_date].copy()
df_series = df_series.reset_index(drop=True)

y = df_series['Sales'].values.astype(float)
dates = pd.to_datetime(df_series['Date']).dt.to_period('Q')

print(f"  [OK] Loaded {len(df_series)} observations")
print(f"  Period: 2012 Q3 - {df_series['Date'].iloc[-1].date()}")
print(f"  Sales: {y.min():,.0f} to {y.max():,.0f}, Mean: {y.mean():,.0f}")

# ============================================================================
# STATIONARITY TESTING
# ============================================================================
print("\n[Step 2] Stationarity analysis...")

adf_result = adfuller(y, autolag='AIC')
print(f"  ADF Test: p-value = {adf_result[1]:.4f}")
print(f"  Result: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")

# ============================================================================
# ACF/PACF PLOTS FOR DIAGNOSIS
# ============================================================================
print("\n[Step 3] Generating ACF/PACF diagnostic plots...")

fig, axes = plt.subplots(3, 2, figsize=(16, 15))

nlags = min(20, len(y) // 2 - 1)

# Original series
plot_acf(y, lags=nlags, ax=axes[0, 0], title='ACF - Original Series (d=0)')
axes[0, 0].set_xlabel('Lag')
plot_pacf(y, lags=nlags, ax=axes[0, 1], title='PACF - Original Series (d=0)', method='ywmle')
axes[0, 1].set_xlabel('Lag')

# First difference
y_diff1 = np.diff(y)
plot_acf(y_diff1, lags=nlags, ax=axes[1, 0], title='ACF - First Difference (d=1)')
axes[1, 0].set_xlabel('Lag')
plot_pacf(y_diff1, lags=nlags, ax=axes[1, 1], title='PACF - First Difference (d=1)', method='ywmle')
axes[1, 1].set_xlabel('Lag')

# Second difference
y_diff2 = np.diff(y_diff1)
plot_acf(y_diff2, lags=nlags, ax=axes[2, 0], title='ACF - Second Difference (d=2)')
axes[2, 0].set_xlabel('Lag')
plot_pacf(y_diff2, lags=nlags, ax=axes[2, 1], title='PACF - Second Difference (d=2)', method='ywmle')
axes[2, 1].set_xlabel('Lag')

plt.tight_layout()
plt.savefig('acf_pacf_analysis.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: acf_pacf_analysis.png")
plt.close()

# ============================================================================
# MODEL SELECTION WITH GRID SEARCH
# ============================================================================
print("\n[Step 4] Model selection (grid search)...")

best_aic = np.inf
best_model_spec = (0, 0, 0)
results_list = []

for d in [0, 1, 2]:
    for p in range(0, 6):
        for q in range(0, 6):
            try:
                model = ARIMA(y, order=(p, d, q))
                fitted = model.fit()
                results_list.append({
                    'order': (p, d, q),
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                    'rmse': np.sqrt(fitted.mse)
                })
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model_spec = (p, d, q)
            except:
                continue

results_df = pd.DataFrame(results_list).sort_values('aic')
print(f"  Best model: ARIMA{best_model_spec}")
print(f"  AIC: {best_aic:.2f}")
print("\n  Top 5 models:")
for idx, row in results_df.head(5).iterrows():
    print(f"    ARIMA{row['order']}: AIC={row['aic']:.2f}")

# ============================================================================
# FIT FINAL MODEL
# ============================================================================
print(f"\n[Step 5] Fitting ARIMA{best_model_spec} model...")

arima_final = ARIMA(y, order=best_model_spec)
arima_final_fitted = arima_final.fit()

print(f"  AIC: {arima_final_fitted.aic:.2f}")
print(f"  BIC: {arima_final_fitted.bic:.2f}")
print(f"  RMSE: {np.sqrt(arima_final_fitted.mse):,.0f}")

# ============================================================================
# OUT-OF-SAMPLE VALIDATION
# ============================================================================
print("\n[Step 6] Out-of-sample validation...")

train_end_idx = len(df_series) - 4
y_train = y[:train_end_idx]
y_test = y[train_end_idx:]
dates_train = dates[:train_end_idx]
dates_test = dates[train_end_idx:]

arima_train = ARIMA(y_train, order=best_model_spec)
arima_train_fitted = arima_train.fit()

forecast_test = arima_train_fitted.get_forecast(steps=len(y_test))
y_test_pred = np.array(forecast_test.predicted_mean)
pred_intervals_test = np.array(forecast_test.conf_int(alpha=0.05))

test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
print(f"  Test MAPE: {test_mape:.2f}%")

# ============================================================================
# FUTURE FORECASTING
# ============================================================================
print("\n[Step 7] Generating future forecasts...")

forecast_all = arima_final_fitted.get_forecast(steps=6)
forecast_values = np.array(forecast_all.predicted_mean)
forecast_intervals = np.array(forecast_all.conf_int(alpha=0.05))

print(f"  2025 Q4: {forecast_values[4]:,.0f} [CI: {forecast_intervals[4,0]:,.0f} - {forecast_intervals[4,1]:,.0f}]")
print(f"  2026 Q1: {forecast_values[5]:,.0f} [CI: {forecast_intervals[5,0]:,.0f} - {forecast_intervals[5,1]:,.0f}]")

# ============================================================================
# CREATE RESULTS CSV
# ============================================================================
print("\n[Step 8] Creating results CSV...")

results_data = []

# Test period
for i, (date, actual, pred, ci_lower, ci_upper) in enumerate(
    zip(dates_test, y_test, y_test_pred, pred_intervals_test[:, 0], pred_intervals_test[:, 1])
):
    results_data.append({
        'Date': date.to_timestamp().strftime('%Y-%m-%d'),
        'Year': date.year,
        'Quarter': date.quarter,
        'Actual_Sales': actual,
        'Forecast_Sales': pred,
        'Error': actual - pred,
        'Error_Pct': (actual - pred) / actual * 100,
        'CI_Lower_95': ci_lower,
        'CI_Upper_95': ci_upper,
        'CI_Width': ci_upper - ci_lower
    })

# Future forecasts
last_test_date = dates_test.iloc[-1]
future_dates = pd.period_range(start=last_test_date + 1, periods=2, freq='Q')

for i, (date, pred, ci_lower, ci_upper) in enumerate(
    zip(future_dates, forecast_values[4:6], forecast_intervals[4:6, 0], forecast_intervals[4:6, 1])
):
    results_data.append({
        'Date': date.to_timestamp().strftime('%Y-%m-%d'),
        'Year': date.year,
        'Quarter': date.quarter,
        'Actual_Sales': np.nan,
        'Forecast_Sales': pred,
        'Error': np.nan,
        'Error_Pct': np.nan,
        'CI_Lower_95': ci_lower,
        'CI_Upper_95': ci_upper,
        'CI_Width': ci_upper - ci_lower
    })

forecast_results_df = pd.DataFrame(results_data)
forecast_results_df.to_csv('arma_forecast_results.csv', index=False)
print("  [OK] Saved: arma_forecast_results.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[Step 9] Creating visualizations...")

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.35)

# Convert dates for plotting
dates_all = pd.to_datetime([d.to_timestamp() for d in dates])
dates_test_plot = pd.to_datetime([d.to_timestamp() for d in dates_test])

# 1. Time series
ax1 = fig.add_subplot(gs[0, 0])
y_fitted = arima_train_fitted.fittedvalues

ax1.plot(dates_all[:train_end_idx], y_train, 'o-', label='Actual (Train)', color='#2E86AB',
         markersize=5, linewidth=2, alpha=0.8)
ax1.plot(dates_all[:train_end_idx], y_fitted, '--', label='Fitted (Train)', 
         color='#06D6A0', linewidth=2, alpha=0.7)
ax1.plot(dates_test_plot, y_test, 'o-', label='Actual (Test)', color='#2E86AB',
         markersize=8, linewidth=2.5)
ax1.plot(dates_test_plot, y_test_pred, 's--', label='Forecast (Test)', color='#EF476F',
         markersize=8, linewidth=2.5)

ax1.fill_between(dates_test_plot, pred_intervals_test[:, 0], pred_intervals_test[:, 1],
                  color='#EF476F', alpha=0.2, label='95% CI')

future_dates_plot = [pd.Timestamp('2025-12-31'), pd.Timestamp('2026-03-31')]
ax1.plot(future_dates_plot, forecast_values[4:6], 'D--', color='#FF6B35', markersize=10,
         linewidth=2.5, label='Future Forecast')
ax1.fill_between(future_dates_plot, forecast_intervals[4:6, 0], forecast_intervals[4:6, 1],
                  color='#FF6B35', alpha=0.2)

ax1.axvline(x=dates_test_plot[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quarterly Sales', fontsize=12, fontweight='bold')
ax1.set_title('ARMA Forecast: Actual vs Predicted Over Time',
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='best', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Residuals vs Fitted
ax2 = fig.add_subplot(gs[0, 1])
residuals_train = y_train - y_fitted
ax2.scatter(y_fitted, residuals_train, alpha=0.6, s=80, color='#118AB2',
            edgecolors='black', linewidth=0.8)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2.5, alpha=0.8)
ax2.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax2.set_title('Residuals vs Fitted (Training Set)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.suptitle(f'ARMA Forecasting Model - ARIMA{best_model_spec} (Box-Jenkins)',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('arma_forecast_analysis.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: arma_forecast_analysis.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"\nGenerated Files:")
print(f"  1. arma_forecast_results.csv - Forecasts with 95% CIs")
print(f"  2. arma_forecast_analysis.png - Time series and residuals")
print(f"  3. acf_pacf_analysis.png - ACF/PACF diagnostic plots")
print(f"  4. arma_forecast_final.py - This script")

print(f"\nModel Summary:")
print(f"  Model: ARIMA{best_model_spec}")
print(f"  AIC: {arima_final_fitted.aic:.2f}")
print(f"  Test MAPE: {test_mape:.2f}%")
print(f"  Data period: 2012 Q3 - 2025 Q3 ({len(y_train)} training obs)")

print(f"\n" + "="*80)
print("[OK] All outputs saved to Q5 folder")
print("="*80 + "\n")

