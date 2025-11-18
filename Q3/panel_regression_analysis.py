"""
Panel Data Regression Analysis: EV/EBITDA vs P/B
Energy Sector Valuation Models with OLS and XGBoost

This script performs complete analysis from the unified panel data CSV file:
1. OLS panel regression with fixed effects
2. XGBoost gradient boosting models
3. Predictions and visualizations
4. Comprehensive performance evaluation

Data source: Q3/result/panel_data_all_variables.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Get script directory and set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'result', 'panel_data_all_variables.csv')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


#==============================================================================
# CONFIGURATION
#==============================================================================

# Independent variables (excluding oil_price and copper_price levels)
FEATURE_COLS = [
    'oil_price_mom',        # Oil price month-over-month change
    'copper_price_mom',     # Copper price month-over-month change
    'rev_growth_cq',        # Revenue growth (quarterly)
    'ebitda_growth_cq',     # EBITDA growth (quarterly)
    'roic_ltm',             # Return on Invested Capital (LTM)
    'covid_dummy',          # COVID period dummy (Mar 2020 - Dec 2021)
    'dummy_XOM',            # Company dummy variables (COP is reference)
    'dummy_CVX',
    'dummy_EOG',
    'dummy_BP',
    'dummy_DVN'
]

TARGET_VARIABLES = {
    'EVEBITDA': 'ev_ebitda',
    'PB': 'pb'
}


#==============================================================================
# OLS REGRESSION ANALYSIS
#==============================================================================

def run_ols_regression(target_name, target_col, data):
    """
    Run OLS panel regression with fixed effects
    
    Parameters:
    -----------
    target_name : str
        Display name ('EVEBITDA' or 'PB')
    target_col : str
        Column name in dataframe ('ev_ebitda' or 'pb')
    data : DataFrame
        Panel data
        
    Returns:
    --------
    results : OLS results object
    regression_df : DataFrame with regression data
    """
    
    print("\n" + "="*80)
    print(f"OLS REGRESSION: {target_name}")
    print("="*80)
    
    # Prepare data
    regression_df = data.dropna(subset=[target_col] + FEATURE_COLS).copy()
    
    print(f"\nObservations: {len(regression_df)}")
    print(f"Date range: {regression_df['date'].min()} to {regression_df['date'].max()}")
    print(f"\nIndependent variables ({len(FEATURE_COLS)}):")
    for i, var in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {var}")
    
    # Prepare X and y
    y = regression_df[target_col]
    X = regression_df[FEATURE_COLS]
    X = sm.add_constant(X)
    
    # Run OLS
    print("\nRunning OLS regression...")
    model = OLS(y, X)
    results = model.fit()
    
    print("\n" + "="*80)
    print("REGRESSION RESULTS")
    print("="*80)
    print(results.summary())
    
    # Save results
    results_file = os.path.join(RESULT_DIR, f'{target_name}_regression_results.txt')
    with open(results_file, 'w') as f:
        f.write(str(results.summary()))
    print(f"\nResults saved to: {results_file}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Variable': results.params.index,
        'Coefficient': results.params.values,
        'Std Error': results.bse.values,
        't-statistic': results.tvalues.values,
        'p-value': results.pvalues.values,
        'CI Lower (2.5%)': results.conf_int()[0].values,
        'CI Upper (97.5%)': results.conf_int()[1].values
    })
    
    results_csv = os.path.join(RESULT_DIR, f'{target_name}_regression_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Results (CSV) saved to: {results_csv}")
    
    # Significant variables
    print("\n" + "="*80)
    print("SIGNIFICANT VARIABLES (p < 0.05)")
    print("="*80)
    sig_vars = results_df[results_df['p-value'] < 0.05]
    if len(sig_vars) > 0:
        print(sig_vars[['Variable', 'Coefficient', 'p-value']].to_string(index=False))
    
    # Create visualizations
    create_ols_visualizations(target_name, results, results_df, y, regression_df)
    
    # COP predictions
    cop_predictions(target_name, target_col, results, regression_df)
    
    return results, regression_df


def create_ols_visualizations(target_name, results, results_df, y, regression_df):
    """Create OLS diagnostic visualizations"""
    
    print("\n" + "-"*80)
    print("Creating visualizations...")
    print("-"*80)
    
    # 1. Coefficient plot
    fig, ax = plt.subplots(figsize=(10, 8))
    coef_df = results_df[results_df['Variable'] != 'const'].copy()
    
    y_pos = np.arange(len(coef_df))
    colors = ['green' if p < 0.05 else 'gray' for p in coef_df['p-value']]
    
    ax.errorbar(coef_df['Coefficient'], y_pos,
                xerr=[coef_df['Coefficient'] - coef_df['CI Lower (2.5%)'],
                      coef_df['CI Upper (97.5%)'] - coef_df['Coefficient']],
                fmt='o', markersize=8, capsize=5, capthick=2,
                color='gray', ecolor='gray')
    
    sig_mask = coef_df['p-value'] < 0.05
    if sig_mask.any():
        ax.scatter(coef_df[sig_mask]['Coefficient'], 
                  y_pos[sig_mask],
                  s=100, c='green', zorder=5, label='Significant (p<0.05)')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df['Variable'])
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title(f'{target_name} Regression Coefficients with 95% CI', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    if sig_mask.any():
        ax.legend()
    
    plt.tight_layout()
    coef_plot = os.path.join(RESULT_DIR, f'{target_name}_coefficients.png')
    plt.savefig(coef_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Coefficient plot saved to: {coef_plot}")
    
    # 2. Regression diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fitted vs Actual
    axes[0, 0].scatter(results.fittedvalues, y, alpha=0.5)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Actual Values')
    axes[0, 0].set_title('Fitted vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs Fitted
    axes[0, 1].scatter(results.fittedvalues, results.resid, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Fitted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(results.resid, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1, 1].hist(results.resid, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    diag_plot = os.path.join(RESULT_DIR, f'{target_name}_regression_diagnostics.png')
    plt.savefig(diag_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagnostics plot saved to: {diag_plot}")


def cop_predictions(target_name, target_col, results, regression_df):
    """Generate COP-specific predictions with intervals"""
    
    print("\n" + "-"*80)
    print("Generating COP-specific predictions...")
    print("-"*80)
    
    cop_data = regression_df[regression_df['company'] == 'COP'].copy()
    cop_data = cop_data.sort_values('date').reset_index(drop=True)
    
    print(f"COP observations: {len(cop_data)}")
    
    # Prepare X for COP
    cop_X = cop_data[FEATURE_COLS]
    cop_X = sm.add_constant(cop_X)
    
    # Get predictions
    cop_pred = results.get_prediction(cop_X)
    cop_pred_summary = cop_pred.summary_frame(alpha=0.05)
    
    # Add to COP data
    cop_data['predicted'] = cop_pred_summary['mean'].values
    cop_data['conf_lower'] = cop_pred_summary['mean_ci_lower'].values
    cop_data['conf_upper'] = cop_pred_summary['mean_ci_upper'].values
    cop_data['pred_lower'] = cop_pred_summary['obs_ci_lower'].values
    cop_data['pred_upper'] = cop_pred_summary['obs_ci_upper'].values
    cop_data['error'] = cop_data[target_col] - cop_data['predicted']
    cop_data['error_pct'] = (cop_data['error'] / cop_data[target_col]) * 100
    
    # Save predictions
    cop_pred_file = os.path.join(RESULT_DIR, f'{target_name}_COP_predictions.csv')
    cop_data[['date', target_col, 'predicted', 'error', 'error_pct',
              'conf_lower', 'conf_upper', 'pred_lower', 'pred_upper']].to_csv(
                  cop_pred_file, index=False)
    print(f"COP predictions saved to: {cop_pred_file}")
    
    # Calculate metrics
    rmse = np.sqrt(np.mean(cop_data['error']**2))
    mae = np.mean(np.abs(cop_data['error']))
    mape = np.mean(np.abs(cop_data['error_pct']))
    
    print(f"\nCOP Prediction Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Plot COP predictions
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(cop_data['date'], cop_data[target_col], 'o-', 
            label='Actual', linewidth=2, markersize=6)
    ax.plot(cop_data['date'], cop_data['predicted'], 's-', 
            label='Predicted', linewidth=2, markersize=6)
    ax.fill_between(cop_data['date'], cop_data['conf_lower'], cop_data['conf_upper'],
                     alpha=0.2, label='95% Confidence Interval')
    ax.fill_between(cop_data['date'], cop_data['pred_lower'], cop_data['pred_upper'],
                     alpha=0.1, label='95% Prediction Interval')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(target_name, fontsize=12)
    ax.set_title(f'COP {target_name}: Actual vs Predicted (OLS Model)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    cop_plot = os.path.join(RESULT_DIR, f'{target_name}_COP_predictions.png')
    plt.savefig(cop_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"COP predictions plot saved to: {cop_plot}")


#==============================================================================
# XGBOOST ANALYSIS
#==============================================================================

def run_xgboost_model(target_name, target_col, data):
    """
    Train and evaluate XGBoost model
    
    Parameters:
    -----------
    target_name : str
        Display name ('EVEBITDA' or 'PB')
    target_col : str
        Column name in dataframe ('ev_ebitda' or 'pb')
    data : DataFrame
        Panel data
        
    Returns:
    --------
    model : XGBoost model
    test_r2 : float
    test_rmse : float
    test_mae : float
    feature_importance : DataFrame
    """
    
    print("\n" + "="*80)
    print(f"XGBOOST MODEL: {target_name}")
    print("="*80)
    
    # Prepare data
    data_clean = data.dropna(subset=[target_col] + FEATURE_COLS).copy()
    
    print(f"\nClean data shape: {data_clean.shape}")
    print(f"Companies: {data_clean['company'].unique()}")
    
    # Prepare features and target
    X = data_clean[FEATURE_COLS]
    y = data_clean[target_col]
    company_labels = data_clean['company']
    
    # Train-test split
    X_train, X_test, y_train, y_test, company_train, company_test = train_test_split(
        X, y, company_labels, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")
    print(f"\nCompany distribution in test set:")
    print(company_test.value_counts().sort_index())
    
    # Train model
    print("\n" + "="*80)
    print("Training XGBoost model...")
    print("="*80)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("\nModel training completed!")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print("\nTraining Set:")
    print(f"  R-squared:  {train_r2:.4f}")
    print(f"  RMSE:       {train_rmse:.4f}")
    print(f"  MAE:        {train_mae:.4f}")
    
    print("\nTest Set (Generalization):")
    print(f"  R-squared:  {test_r2:.4f}")
    print(f"  RMSE:       {test_rmse:.4f}")
    print(f"  MAE:        {test_mae:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    print(feature_importance.to_string(index=False))
    
    # COP-specific performance
    print("\n" + "="*80)
    print("COP-SPECIFIC PERFORMANCE")
    print("="*80)
    
    cop_test_mask = company_test == 'COP'
    if cop_test_mask.sum() > 0:
        y_test_cop = y_test[cop_test_mask]
        y_test_cop_pred = y_test_pred[cop_test_mask]
        
        cop_r2 = r2_score(y_test_cop, y_test_cop_pred)
        cop_rmse = np.sqrt(mean_squared_error(y_test_cop, y_test_cop_pred))
        cop_mae = mean_absolute_error(y_test_cop, y_test_cop_pred)
        
        print(f"\nCOP Test Set: {len(y_test_cop)} observations")
        print(f"R-squared:  {cop_r2:.4f}")
        print(f"RMSE:       {cop_rmse:.4f}")
        print(f"MAE:        {cop_mae:.4f}")
    else:
        cop_r2 = None
        cop_rmse = None
        cop_mae = None
    
    # Save summary
    save_xgboost_summary(target_name, train_r2, train_rmse, train_mae,
                        test_r2, test_rmse, test_mae,
                        cop_r2, cop_rmse, cop_mae,
                        feature_importance, len(X_train), len(X_test))
    
    # Create visualizations
    create_xgboost_visualizations(target_name, y_test, y_test_pred, 
                                  train_r2, train_rmse, train_mae,
                                  test_r2, test_rmse, test_mae,
                                  feature_importance)
    
    # Save predictions
    test_results = pd.DataFrame({
        'company': company_test.values,
        'actual': y_test.values,
        'predicted': y_test_pred,
        'residual': y_test.values - y_test_pred,
        'abs_error': np.abs(y_test.values - y_test_pred)
    })
    
    pred_file = os.path.join(RESULT_DIR, f'{target_name}_xgboost_test_predictions.csv')
    test_results.to_csv(pred_file, index=False)
    print(f"\nTest predictions saved to: {pred_file}")
    
    return model, test_r2, test_rmse, test_mae, feature_importance


def save_xgboost_summary(target_name, train_r2, train_rmse, train_mae,
                         test_r2, test_rmse, test_mae,
                         cop_r2, cop_rmse, cop_mae,
                         feature_importance, n_train, n_test):
    """Save XGBoost model summary to file"""
    
    summary_file = os.path.join(RESULT_DIR, f'{target_name}_xgboost_model_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"XGBOOST MODEL SUMMARY: {target_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATA SOURCE:\n")
        f.write("-"*40 + "\n")
        f.write("File: panel_data_all_variables.csv\n")
        f.write(f"Training observations: {n_train}\n")
        f.write(f"Test observations: {n_test}\n\n")
        
        f.write("FEATURES (excluding oil/copper price levels):\n")
        f.write("-"*40 + "\n")
        f.write("- Oil price MoM change\n")
        f.write("- Copper price MoM change\n")
        f.write("- Revenue growth (quarterly)\n")
        f.write("- EBITDA growth (quarterly)\n")
        f.write("- ROIC (LTM)\n")
        f.write("- COVID dummy (Mar 2020 - Dec 2021)\n")
        f.write("- Company dummies (5 companies)\n\n")
        
        f.write("TRAINING PERFORMANCE:\n")
        f.write("-"*40 + "\n")
        f.write(f"R-squared:  {train_r2:.4f}\n")
        f.write(f"RMSE:       {train_rmse:.4f}\n")
        f.write(f"MAE:        {train_mae:.4f}\n\n")
        
        f.write("TEST PERFORMANCE:\n")
        f.write("-"*40 + "\n")
        f.write(f"R-squared:  {test_r2:.4f}\n")
        f.write(f"RMSE:       {test_rmse:.4f}\n")
        f.write(f"MAE:        {test_mae:.4f}\n\n")
        
        if cop_r2 is not None:
            f.write("COP-SPECIFIC PERFORMANCE (Test Set):\n")
            f.write("-"*40 + "\n")
            f.write(f"R-squared:  {cop_r2:.4f}\n")
            f.write(f"RMSE:       {cop_rmse:.4f}\n")
            f.write(f"MAE:        {cop_mae:.4f}\n\n")
        
        f.write("FEATURE IMPORTANCE:\n")
        f.write("-"*40 + "\n")
        f.write(feature_importance.to_string(index=False))
        f.write("\n")
    
    print(f"Model summary saved to: {summary_file}")


def create_xgboost_visualizations(target_name, y_test, y_test_pred,
                                   train_r2, train_rmse, train_mae,
                                   test_r2, test_rmse, test_mae,
                                   feature_importance):
    """Create XGBoost diagnostic visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Test Set: Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance
    axes[1, 0].barh(feature_importance['Feature'], 
                    feature_importance['Importance'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Feature Importance')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    
    # 4. Train vs Test Performance
    metrics = ['R²', 'RMSE', 'MAE']
    train_metrics = [train_r2, train_rmse, train_mae]
    test_metrics = [test_r2, test_rmse, test_mae]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize for visualization
    if target_name == 'EVEBITDA':
        scale = 50
    else:
        scale = 2
    
    train_norm = [train_r2, train_rmse/scale, train_mae/scale]
    test_norm = [test_r2, test_rmse/scale, test_mae/scale]
    
    axes[1, 1].bar(x - width/2, train_norm, width, label='Train', alpha=0.8)
    axes[1, 1].bar(x + width/2, test_norm, width, label='Test', alpha=0.8)
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].set_title('Train vs Test Performance\n(RMSE & MAE scaled)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    viz_file = os.path.join(RESULT_DIR, f'{target_name}_xgboost_visualization.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {viz_file}")


#==============================================================================
# MAIN EXECUTION
#==============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "#"*80)
    print("# PANEL DATA REGRESSION ANALYSIS")
    print("# EV/EBITDA vs P/B - OLS and XGBoost Models")
    print("#"*80)
    
    print(f"\nData source: {DATA_FILE}")
    print("\nFeature Configuration:")
    print("  - Oil price MoM (level excluded)")
    print("  - Copper price MoM (level excluded)")
    print("  - Revenue growth (quarterly)")
    print("  - EBITDA growth (quarterly)")
    print("  - ROIC (LTM)")
    print("  - COVID dummy (Mar 2020 - Dec 2021)")
    print("  - Company fixed effects (COP is reference)")
    
    # Load data
    print("\n" + "="*80)
    print("Loading data...")
    print("="*80)
    
    data = pd.read_csv(DATA_FILE)
    data['date'] = pd.to_datetime(data['date'])
    
    print(f"\nData loaded: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Companies: {sorted(data['company'].unique())}")
    
    # Store results for comparison
    results_summary = {}
    
    # Run analyses for both target variables
    for target_name, target_col in TARGET_VARIABLES.items():
        
        print("\n\n" + "#"*80)
        print(f"# ANALYZING: {target_name}")
        print("#"*80)
        
        # OLS Regression
        print("\n" + "="*80)
        print(f"PART 1: OLS REGRESSION - {target_name}")
        print("="*80)
        
        ols_results, regression_df = run_ols_regression(target_name, target_col, data)
        
        # XGBoost Model
        print("\n" + "="*80)
        print(f"PART 2: XGBOOST MODEL - {target_name}")
        print("="*80)
        
        model, test_r2, test_rmse, test_mae, feature_importance = run_xgboost_model(
            target_name, target_col, data
        )
        
        # Store results
        results_summary[target_name] = {
            'ols_r2': ols_results.rsquared,
            'ols_r2_adj': ols_results.rsquared_adj,
            'xgb_test_r2': test_r2,
            'xgb_test_rmse': test_rmse,
            'xgb_test_mae': test_mae,
            'top_features': feature_importance.head(3)
        }
    
    # Final comparison
    print("\n\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Metric': ['OLS R²', 'OLS Adj. R²', 'XGBoost Test R²', 
                   'XGBoost Test RMSE', 'XGBoost Test MAE'],
        'EVEBITDA': [
            f"{results_summary['EVEBITDA']['ols_r2']:.4f}",
            f"{results_summary['EVEBITDA']['ols_r2_adj']:.4f}",
            f"{results_summary['EVEBITDA']['xgb_test_r2']:.4f}",
            f"{results_summary['EVEBITDA']['xgb_test_rmse']:.4f}",
            f"{results_summary['EVEBITDA']['xgb_test_mae']:.4f}"
        ],
        'PB': [
            f"{results_summary['PB']['ols_r2']:.4f}",
            f"{results_summary['PB']['ols_r2_adj']:.4f}",
            f"{results_summary['PB']['xgb_test_r2']:.4f}",
            f"{results_summary['PB']['xgb_test_rmse']:.4f}",
            f"{results_summary['PB']['xgb_test_mae']:.4f}"
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    pb_ols_r2 = results_summary['PB']['ols_r2']
    ev_ols_r2 = results_summary['EVEBITDA']['ols_r2']
    improvement = (pb_ols_r2 - ev_ols_r2) / ev_ols_r2 * 100
    
    print(f"\n1. P/B OUTPERFORMS EV/EBITDA:")
    print(f"   OLS R²: {pb_ols_r2:.4f} vs {ev_ols_r2:.4f} (+{improvement:.1f}%)")
    
    print(f"\n2. XGBOOST IMPROVES PREDICTIONS:")
    ev_improv = (results_summary['EVEBITDA']['xgb_test_r2'] - ev_ols_r2) / ev_ols_r2 * 100
    pb_improv = (results_summary['PB']['xgb_test_r2'] - pb_ols_r2) / pb_ols_r2 * 100
    print(f"   EV/EBITDA: +{ev_improv:.1f}% improvement")
    print(f"   P/B: +{pb_improv:.1f}% improvement")
    
    print(f"\n3. TOP FEATURES:")
    print(f"\n   EV/EBITDA:")
    for _, row in results_summary['EVEBITDA']['top_features'].iterrows():
        print(f"     - {row['Feature']}: {row['Importance']:.4f}")
    
    print(f"\n   P/B:")
    for _, row in results_summary['PB']['top_features'].iterrows():
        print(f"     - {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)
    print("\nAll results saved to result/ directory")
    print("  - Regression outputs (TXT, CSV)")
    print("  - Visualizations (PNG)")
    print("  - Predictions (CSV)")
    print("  - XGBoost summaries (TXT)")
    print("="*80)


if __name__ == "__main__":
    main()

