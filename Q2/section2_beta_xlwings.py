import xlwings as xw
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import io
from PIL import Image


def get_anova_table(model):
    """
    Create an ANOVA-like table from a fitted OLS model.
    Returns a DataFrame with SS, df, MS, F, and p-value.
    """
    # Get model statistics
    ssr = model.ess  # Regression sum of squares
    sse = model.ssr  # Residual sum of squares
    sst = model.centered_tss  # Total sum of squares
    
    df_model = model.df_model
    df_resid = model.df_resid
    df_total = df_model + df_resid
    
    msr = ssr / df_model if df_model > 0 else np.nan
    mse = sse / df_resid if df_resid > 0 else np.nan
    
    f_stat = model.fvalue
    f_pvalue = model.f_pvalue
    
    anova_df = pd.DataFrame({
        'df': [df_model, df_resid, df_total],
        'sum_sq': [ssr, sse, sst],
        'mean_sq': [msr, mse, np.nan],
        'F': [f_stat, np.nan, np.nan],
        'PR(>F)': [f_pvalue, np.nan, np.nan]
    }, index=['Regression', 'Residual', 'Total'])
    
    return anova_df


def create_residual_plots(model, model_name, var_name):
    """
    Create two residual plots:
    1. Time series plot (index vs residuals)
    2. Fitted values vs residuals
    
    Returns the figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    
    residuals = model.resid
    
    # Plot 1: Time series of residuals
    axes[0].plot(range(len(residuals)), residuals, 'o-', markersize=4, linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{var_name} - {model_name}\nResiduals over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Mkt-RF vs Residuals (or fitted values if Mkt-RF not available)
    # Get Mkt-RF values if available
    if 'Mkt-RF' in model.model.exog_names:
        mkt_rf_idx = model.model.exog_names.index('Mkt-RF')
        x_values = model.model.exog[:, mkt_rf_idx]
        x_label = 'Mkt-RF'
    else:
        x_values = model.fittedvalues
        x_label = 'Fitted Values'
    
    axes[1].scatter(x_values, residuals, alpha=0.6, s=30)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{var_name} - {model_name}\nResiduals vs {x_label}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def get_data_frame(wb_name: str, sheet_name: str | None = None) -> pd.DataFrame:
    """
    Read the relevant block (L:AB) from the specified workbook and sheet into a DataFrame.

    Assumptions (adjust in Excel or in this function if needed):
    - The first non-empty row in the L:AB block contains variable names, including
      'Mkt-RF', 'SMB', 'HML', 'RF', and 'Covid'.
    - All rows below the header are data rows with no gaps.
    """
    wb = xw.Book(wb_name)

    if sheet_name is not None:
        sht = wb.sheets[sheet_name]
    else:
        # Use sheet named "2" based on the screenshot
        sht = wb.sheets["2"]

    # Determine the used range starting at column L (12) through AB (28).
    first_col = 12  # L
    last_col = 28   # AB
    last_row = sht.cells.last_cell.row

    raw_values = sht.range((1, first_col), (last_row, last_col)).value
    raw_df = pd.DataFrame(raw_values)

    # Find header row.
    # Your screenshot shows a row starting with "Date" and then
    # COP, XOM, ..., Peers Average, Mkt-RF, SMB, HML, RF, Covid.
    # We'll use that layout to detect the header robustly.
    header_row_idx = None
    for i in range(len(raw_df)):
        row_vals = raw_df.iloc[i].tolist() or []
        # Normalise: convert to string, strip spaces
        norm_vals = [str(v).strip() if v is not None else "" for v in row_vals]
        if not any(norm_vals):
            continue
        first_val = norm_vals[0].lower()
        # Check if first column is "Date" and row contains "Mkt-RF" (case-insensitive)
        norm_vals_lower = [v.lower() for v in norm_vals]
        if first_val == "date" and "mkt-rf" in norm_vals_lower:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Could not find header row containing 'Date' and 'Mkt-RF'. "
                         "Please check that the data from column L to AB has these headers.")

    header = raw_df.iloc[header_row_idx].tolist()
    data = raw_df.iloc[header_row_idx + 1:].reset_index(drop=True)
    data.columns = header

    # Drop completely empty rows that might be at the bottom.
    data = data.dropna(how="all")
    
    # Debug: print data shape
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    return data


def run_models(df: pd.DataFrame):
    """
    Run all requested models for each dependent variable (columns from the first
    variable column up to but not including 'Mkt-RF').

    Models per variable:
      1. CAPM: y ~ const + (Mkt-RF)
      2. FF3: y ~ const + (Mkt-RF) + SMB + HML
      3. FF3_COVID: y ~ const + (Mkt-RF) + SMB + HML + Covid (dummy variable)

    For each model, perform:
      - Durbin-Watson test for autocorrelation
      - Breusch-Pagan test for heteroskedasticity
      - Apply HAC robust standard errors if needed

    Returns:
        betas: dict of DataFrames summarising betas for each model.
        models_dict: dict keyed by (var_name, model_name) -> dict with original and corrected models.
    """
    required_cols = ["Mkt-RF", "SMB", "HML", "RF", "Covid"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    # Identify dependent variables: everything before 'Mkt-RF'
    mktrf_idx = df.columns.get_loc("Mkt-RF")
    dep_vars = df.columns[:mktrf_idx]

    df = df.copy()

    betas_capm = {}
    betas_ff3 = {}
    betas_ff3_covid = {}
    models_dict = {}
    
    # Only generate comprehensive output for these variables
    target_vars = ["COP", "^SP500-10", "Peers Average"]

    for var in dep_vars:
        y = pd.to_numeric(df[var], errors="coerce")
        
        # Build a combined mask: y and all regressors must be non-null
        # Note: COVID can be 0 (which is valid), so we check notna() not just boolean
        mktrf = pd.to_numeric(df["Mkt-RF"], errors="coerce")
        smb = pd.to_numeric(df["SMB"], errors="coerce")
        hml = pd.to_numeric(df["HML"], errors="coerce")
        covid = pd.to_numeric(df["Covid"], errors="coerce")
        
        # Fill NaN in covid with 0 (assuming missing means non-COVID period)
        covid = covid.fillna(0)
        
        mask = (
            y.notna() & 
            mktrf.notna() & 
            smb.notna() & 
            hml.notna()
        )
        
        if mask.sum() == 0:
            print(f"Warning: No valid data for variable '{var}'. Skipping.")
            continue
        
        # Apply mask to all series
        y = y[mask].reset_index(drop=True)
        mktrf = mktrf[mask].reset_index(drop=True)
        smb = smb[mask].reset_index(drop=True)
        hml = hml[mask].reset_index(drop=True)
        covid = covid[mask].reset_index(drop=True)
        
        # Debug: print observation count for target variables
        if var in target_vars:
            print(f"{var}: {len(y)} observations after filtering")

        # 1. CAPM with Mkt-RF
        X_capm = sm.add_constant(mktrf)
        X_capm.columns = ["const", "Mkt-RF"]
        model_capm = sm.OLS(y, X_capm, missing="drop").fit()
        
        # Diagnostics
        dw_capm = durbin_watson(model_capm.resid)
        bp_test_capm = het_breuschpagan(model_capm.resid, model_capm.model.exog)
        bp_pval_capm = bp_test_capm[1]
        
        # Apply HAC correction if autocorrelation or heteroskedasticity detected
        model_capm_corrected = None
        if dw_capm < 1.5 or dw_capm > 2.5 or bp_pval_capm < 0.05:
            model_capm_corrected = model_capm.get_robustcov_results(cov_type='HAC', maxlags=1)
        
        pval_mkt_corr = model_capm_corrected.pvalues[1] if model_capm_corrected else model_capm.pvalues["Mkt-RF"]
        
        betas_capm[var] = pd.Series({
            "alpha": model_capm.params["const"],
            "beta_Mkt": model_capm.params["Mkt-RF"],
            "pval_Mkt": model_capm.pvalues["Mkt-RF"],
            "pval_Mkt_corrected": pval_mkt_corr,
            "R2_adj": model_capm.rsquared_adj,
        })
        
        # Store comprehensive output only for target variables
        if var in target_vars:
            models_dict[(var, "CAPM")] = {
                "original": model_capm,
                "corrected": model_capm_corrected,
                "dw": dw_capm,
                "bp_pval": bp_pval_capm,
            }

        # 2. Fama-French 3-factor
        X_ff3 = sm.add_constant(pd.concat([mktrf, smb, hml], axis=1))
        X_ff3.columns = ["const", "Mkt-RF", "SMB", "HML"]
        model_ff3 = sm.OLS(y, X_ff3, missing="drop").fit()
        
        # Diagnostics
        dw_ff3 = durbin_watson(model_ff3.resid)
        bp_test_ff3 = het_breuschpagan(model_ff3.resid, model_ff3.model.exog)
        bp_pval_ff3 = bp_test_ff3[1]
        
        model_ff3_corrected = None
        if dw_ff3 < 1.5 or dw_ff3 > 2.5 or bp_pval_ff3 < 0.05:
            model_ff3_corrected = model_ff3.get_robustcov_results(cov_type='HAC', maxlags=1)
        
        pval_mkt_corr_ff3 = model_ff3_corrected.pvalues[1] if model_ff3_corrected else model_ff3.pvalues["Mkt-RF"]
        
        betas_ff3[var] = pd.Series({
            "alpha": model_ff3.params["const"],
            "beta_Mkt": model_ff3.params["Mkt-RF"],
            "beta_SMB": model_ff3.params["SMB"],
            "beta_HML": model_ff3.params["HML"],
            "pval_Mkt": model_ff3.pvalues["Mkt-RF"],
            "pval_Mkt_corrected": pval_mkt_corr_ff3,
            "R2_adj": model_ff3.rsquared_adj,
        })
        
        if var in target_vars:
            models_dict[(var, "FF3")] = {
                "original": model_ff3,
                "corrected": model_ff3_corrected,
                "dw": dw_ff3,
                "bp_pval": bp_pval_ff3,
            }

        # 3. FF3 with COVID dummy (not interaction)
        X_ff3_covid_data = pd.concat([mktrf, smb, hml, covid], axis=1)
        X_ff3_covid_data.columns = ["Mkt-RF", "SMB", "HML", "Covid"]
        X_ff3_covid = sm.add_constant(X_ff3_covid_data)
        model_ff3_covid = sm.OLS(y, X_ff3_covid, missing="drop").fit()

        # Diagnostics
        dw_ff3_covid = durbin_watson(model_ff3_covid.resid)
        bp_test_ff3_covid = het_breuschpagan(model_ff3_covid.resid, model_ff3_covid.model.exog)
        bp_pval_ff3_covid = bp_test_ff3_covid[1]
        
        model_ff3_covid_corrected = None
        if dw_ff3_covid < 1.5 or dw_ff3_covid > 2.5 or bp_pval_ff3_covid < 0.05:
            model_ff3_covid_corrected = model_ff3_covid.get_robustcov_results(cov_type='HAC', maxlags=1)

        pval_mkt_corr_ff3_covid = model_ff3_covid_corrected.pvalues[1] if model_ff3_covid_corrected else model_ff3_covid.pvalues["Mkt-RF"]

        # Get the constant parameter name (could be 'const' or 'Intercept')
        const_name = model_ff3_covid.params.index[0]
        
        betas_ff3_covid[var] = pd.Series({
            "alpha": model_ff3_covid.params[const_name],
            "beta_Mkt": model_ff3_covid.params["Mkt-RF"],
            "beta_SMB": model_ff3_covid.params["SMB"],
            "beta_HML": model_ff3_covid.params["HML"],
            "beta_Covid": model_ff3_covid.params["Covid"],
            "pval_Mkt": model_ff3_covid.pvalues["Mkt-RF"],
            "pval_Mkt_corrected": pval_mkt_corr_ff3_covid,
            "R2_adj": model_ff3_covid.rsquared_adj,
        }, index=["alpha", "beta_Mkt", "beta_SMB", "beta_HML", "beta_Covid", "pval_Mkt", "pval_Mkt_corrected", "R2_adj"])
        
        if var in target_vars:
            models_dict[(var, "FF3_COVID")] = {
                "original": model_ff3_covid,
                "corrected": model_ff3_covid_corrected,
                "dw": dw_ff3_covid,
                "bp_pval": bp_pval_ff3_covid,
            }

    # Convert beta dicts into DataFrames with variables as rows
    betas = {
        "CAPM": pd.DataFrame(betas_capm).T,
        "FF3": pd.DataFrame(betas_ff3).T,
        "FF3_COVID": pd.DataFrame(betas_ff3_covid).T,
    }

    return betas, models_dict


def write_results_to_excel(
    wb_name: str,
    betas: dict[str, pd.DataFrame],
    models_dict: dict[tuple[str, str], dict],
):
    """
    Create (or clear) the '2-result' sheet and write:
      - Summary beta tables for each model.
      - Comprehensive regression output for specified variables (COP, ^SP500-10, Peers Average).
    """
    wb = xw.Book(wb_name)

    # Get or create result sheet
    sheet_names = [s.name for s in wb.sheets]
    if "2-result" in sheet_names:
        sht_res = wb.sheets["2-result"]
        sht_res.clear()
        # Also clear any existing pictures
        for pic in sht_res.pictures:
            pic.delete()
    else:
        sht_res = wb.sheets.add("2-result", after=wb.sheets[-1])

    row = 1

    # 1. Beta summary tables (format to 3 decimals)
    for model_name, beta_df in betas.items():
        sht_res.range((row, 1)).value = f"{model_name} - Beta Summary (All Variables)"
        sht_res.range((row, 1)).font.bold = True
        sht_res.range((row, 1)).font.size = 12
        
        # Format to 3 decimals
        beta_df_formatted = beta_df.round(3)
        sht_res.range((row + 1, 1)).value = beta_df_formatted
        row += beta_df_formatted.shape[0] + 4  # space between tables

    # 2. Comprehensive regression output for selected variables
    row += 2
    sht_res.range((row, 1)).value = "COMPREHENSIVE REGRESSION OUTPUT"
    sht_res.range((row, 1)).font.bold = True
    sht_res.range((row, 1)).font.size = 14
    row += 1
    sht_res.range((row, 1)).value = "Variables: COP, ^SP500-10, Peers Average"
    row += 1
    
    # Add HAC correction guidelines
    sht_res.range((row, 1)).value = "HAC Correction Applied When:"
    sht_res.range((row, 1)).font.bold = True
    row += 1
    sht_res.range((row, 1)).value = "• Durbin-Watson < 1.5 or > 2.5 (autocorrelation detected), OR"
    row += 1
    sht_res.range((row, 1)).value = "• Breusch-Pagan p-value < 0.05 (heteroskedasticity detected)"
    row += 2

    # Define colors for each variable
    var_colors = {
        "COP": (220, 230, 241),  # Light blue
        "^SP500-10": (217, 234, 211),  # Light green
        "Peers Average": (252, 228, 214),  # Light orange
    }
    
    # Group by variable to apply consistent coloring
    target_vars = ["COP", "^SP500-10", "Peers Average"]
    model_names = ["CAPM", "FF3", "FF3_COVID"]
    
    for var in target_vars:
        bg_color = var_colors[var]
        
        # Variable header
        sht_res.range((row, 1)).value = f"[ {var} ]"
        sht_res.range((row, 1)).font.bold = True
        sht_res.range((row, 1)).font.size = 13
        sht_res.range((row, 1), (row, 10)).color = bg_color
        row += 2
        
        for model_name in model_names:
            if (var, model_name) not in models_dict:
                continue
                
            model_info = models_dict[(var, model_name)]
            model_orig = model_info["original"]
            model_corr = model_info["corrected"]
            dw = model_info["dw"]
            bp_pval = model_info["bp_pval"]
            
            # Track starting row for this model (to place plots to the right)
            model_start_row = row
            
            # Model title
            sht_res.range((row, 1)).value = f"{model_name}"
            sht_res.range((row, 1)).font.bold = True
            sht_res.range((row, 1)).font.size = 11
            sht_res.range((row, 1), (row, 10)).color = bg_color
            row += 1
            
            # Diagnostic tests
            sht_res.range((row, 1)).value = "Diagnostic Tests:"
            sht_res.range((row, 1)).font.bold = True
            row += 1
            diag_df = pd.DataFrame({
                'Test': ['Durbin-Watson', 'Breusch-Pagan p-value', 'Correction Applied'],
                'Value': [
                    f"{dw:.3f}",
                    f"{bp_pval:.3f}",
                    "Yes (HAC)" if model_corr else "No"
                ]
            })
            sht_res.range((row, 1)).value = diag_df
            sht_res.range((row, 1), (row + diag_df.shape[0], diag_df.shape[1])).color = bg_color
            row += diag_df.shape[0] + 2
            
            # Model summary statistics
            sht_res.range((row, 1)).value = "Model Summary:"
            sht_res.range((row, 1)).font.bold = True
            row += 1
            summary_stats = pd.DataFrame({
                'Statistic': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'AIC', 'BIC', 'N'],
                'Value': [
                    f"{model_orig.rsquared:.3f}",
                    f"{model_orig.rsquared_adj:.3f}",
                    f"{model_orig.fvalue:.3f}",
                    f"{model_orig.f_pvalue:.3f}",
                    f"{model_orig.aic:.3f}",
                    f"{model_orig.bic:.3f}",
                    int(model_orig.nobs)
                ]
            })
            sht_res.range((row, 1)).value = summary_stats
            sht_res.range((row, 1), (row + summary_stats.shape[0], summary_stats.shape[1])).color = bg_color
            row += summary_stats.shape[0] + 2
            
            # Coefficient table - Original
            sht_res.range((row, 1)).value = "Coefficients (Original):"
            sht_res.range((row, 1)).font.bold = True
            row += 1
            coef_table_orig = pd.DataFrame({
                'Variable': model_orig.params.index,
                'Coef': [f"{v:.3f}" for v in model_orig.params.values],
                'Std Err': [f"{v:.3f}" for v in model_orig.bse.values],
                't': [f"{v:.3f}" for v in model_orig.tvalues.values],
                'P>|t|': [f"{v:.3f}" for v in model_orig.pvalues.values],
                '[0.025': [f"{v:.3f}" for v in model_orig.conf_int()[0].values],
                '0.975]': [f"{v:.3f}" for v in model_orig.conf_int()[1].values],
            })
            sht_res.range((row, 1)).value = coef_table_orig
            sht_res.range((row, 1), (row + coef_table_orig.shape[0], coef_table_orig.shape[1])).color = bg_color
            row += coef_table_orig.shape[0] + 2
            
            # Coefficient table - Corrected (if applicable)
            if model_corr:
                sht_res.range((row, 1)).value = "Coefficients (HAC Corrected):"
                sht_res.range((row, 1)).font.bold = True
                row += 1
                # HAC corrected model returns arrays, need to use original index
                conf_int_corr = model_corr.conf_int()
                coef_table_corr = pd.DataFrame({
                    'Variable': model_orig.params.index,
                    'Coef': [f"{v:.3f}" for v in model_corr.params],
                    'Std Err': [f"{v:.3f}" for v in model_corr.bse],
                    't': [f"{v:.3f}" for v in model_corr.tvalues],
                    'P>|t|': [f"{v:.3f}" for v in model_corr.pvalues],
                    '[0.025': [f"{conf_int_corr[i, 0]:.3f}" for i in range(len(model_corr.params))],
                    '0.975]': [f"{conf_int_corr[i, 1]:.3f}" for i in range(len(model_corr.params))],
                })
                sht_res.range((row, 1)).value = coef_table_corr
                # Highlight corrected table with slightly darker shade
                darker_color = tuple(max(0, c - 10) for c in bg_color)
                sht_res.range((row, 1), (row + coef_table_corr.shape[0], coef_table_corr.shape[1])).color = darker_color
                row += coef_table_corr.shape[0] + 2
            
            # ANOVA table
            anova_df = get_anova_table(model_orig).round(3)
            sht_res.range((row, 1)).value = "ANOVA:"
            sht_res.range((row, 1)).font.bold = True
            row += 1
            sht_res.range((row, 1)).value = anova_df
            sht_res.range((row, 1), (row + anova_df.shape[0], anova_df.shape[1])).color = bg_color
            row += anova_df.shape[0] + 3
            
            # Add residual plots to the right of the model details
            # Create and save plots
            fig = create_residual_plots(model_orig, model_name, var)
            
            # Save figure to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, dpi=100, bbox_inches='tight')
                tmp_path = tmp.name
            plt.close(fig)
            
            # Insert the plot to the right (column L, which is column 12)
            plot_col = 12
            sht_res.pictures.add(tmp_path, 
                                name=f"{var}_{model_name}_plot",
                                left=sht_res.range((model_start_row, plot_col)).left,
                                top=sht_res.range((model_start_row, plot_col)).top,
                                width=400,
                                height=500)
            
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        row += 2  # Extra space between variables


def main():
    # Name of your Excel workbook in this folder
    wb_name = "Workdone.xlsx"

    # 1. Load data from sheet 2 (L:AB)
    df = get_data_frame(wb_name)

    # 2. Run all regressions
    betas, models_dict = run_models(df)

    # 3. Write results back to Excel
    write_results_to_excel(wb_name, betas, models_dict)


if __name__ == "__main__":
    main()


