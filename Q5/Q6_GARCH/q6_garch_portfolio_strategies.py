import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

from pathlib import Path
output_dir = Path('.')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("QUESTION 6: GARCH MODEL - PORTFOLIO STRATEGY COMPARISON")
print("="*80)

# Load data
print("\nLoading historical data...")
df = pd.read_csv('../Q52.csv', skiprows=7)
cop_returns = df['COP'].dropna().values
spx_returns = df['SPX'].dropna().values

print(f"COP weekly returns: {len(cop_returns)} observations")
print(f"S&P 500 weekly returns: {len(spx_returns)} observations")

# Fit GARCH models
print("\n" + "="*80)
print("FITTING GARCH MODELS")
print("="*80)

print("\nHistorical Mean Returns (for verification):")
print(f"  COP: {(cop_returns.mean()):.6f}%")
print(f"  SPX: {(spx_returns.mean()):.6f}%")

print("\nFitting GARCH(1,1) for COP...")
garch_cop = arch_model(cop_returns * 100, vol='Garch', p=1, q=1)
res_cop = garch_cop.fit(disp='off')
print("COP GARCH Fitted:")
print(f"  Mean: {res_cop.params['mu']:.6f}%")
print(f"  Omega: {res_cop.params['omega']:.8f}")
print(f"  Alpha: {res_cop.params['alpha[1]']:.6f}")
print(f"  Beta: {res_cop.params['beta[1]']:.6f}")

print("\nFitting GARCH(1,1) for S&P 500...")
garch_spx = arch_model(spx_returns * 100, vol='Garch', p=1, q=1)
res_spx = garch_spx.fit(disp='off')
print("S&P 500 GARCH Fitted:")
print(f"  Mean: {res_spx.params['mu']:.6f}%")
print(f"  Omega: {res_spx.params['omega']:.8f}")
print(f"  Alpha: {res_spx.params['alpha[1]']:.6f}")
print(f"  Beta: {res_spx.params['beta[1]']:.6f}")

# IMPORTANT: Use historical means instead of GARCH fitted means
# GARCH fitting can be unstable for mean estimation
print("\nUsing historical means for simulation (more stable than GARCH-fitted means):")

# Simulation parameters
n_simulations = 10000
n_weeks = 52
initial_investment = 100000
cash_weekly_return = 0.01 / 52

# Month structure for rebalancing
month_ends = [5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48, 52]
month_starts = [0] + month_ends[:-1]

# Get GARCH parameters
# Use historical means (more stable) instead of GARCH-fitted means
mu_cop = cop_returns.mean() / 100  # Convert to decimal weekly
print(f"  COP mean: {mu_cop*100:.6f}% weekly = {mu_cop*52*100:.2f}% annualized")

sigma2_cop_init = res_cop.conditional_volatility[-1] ** 2 / 10000
omega_cop = res_cop.params['omega'] / 10000
alpha_cop = res_cop.params['alpha[1]']
beta_cop = res_cop.params['beta[1]']

# Use historical means (more stable) instead of GARCH-fitted means
mu_spx = spx_returns.mean() / 100  # Convert to decimal weekly
print(f"  SPX mean: {mu_spx*100:.6f}% weekly = {mu_spx*52*100:.2f}% annualized")

sigma2_spx_init = res_spx.conditional_volatility[-1] ** 2 / 10000
omega_spx = res_spx.params['omega'] / 10000
alpha_spx = res_spx.params['alpha[1]']
beta_spx = res_spx.params['beta[1]']

print(f"\nGARCH Volatility Parameters:")
print(f"  COP: omega={omega_cop:.8f}, alpha={alpha_cop:.6f}, beta={beta_cop:.6f}")
print(f"  SPX: omega={omega_spx:.8f}, alpha={alpha_spx:.6f}, beta={beta_spx:.6f}")

print("\n" + "="*80)
print("RUNNING MONTE CARLO SIMULATIONS WITH GARCH")
print("="*80)

def garch_simulate(mu, sigma2_init, omega, alpha, beta, n_weeks, rng):
    """Simulate GARCH(1,1) returns"""
    returns = np.zeros(n_weeks)
    sigma2 = np.zeros(n_weeks)
    sigma2[0] = sigma2_init
    
    for t in range(n_weeks):
        z = rng.standard_normal()
        returns[t] = mu + np.sqrt(sigma2[t]) * z
        
        if t < n_weeks - 1:
            sigma2[t + 1] = omega + alpha * (returns[t] ** 2) + beta * sigma2[t]
    
    return returns

# Set random seed
np.random.seed(42)

# ============================================================================
# STRATEGY A: 100% S&P 500
# ============================================================================
print("\nStrategy A: All S&P 500 (GARCH simulated)...")
final_values_a = []

for sim in range(n_simulations):
    rng = np.random.RandomState(seed=sim)
    sampled_spx = garch_simulate(mu_spx, sigma2_spx_init, omega_spx, alpha_spx, beta_spx, n_weeks, rng)
    final_value = initial_investment * np.prod(1 + sampled_spx)
    final_values_a.append(final_value)

final_values_a = np.array(final_values_a)
annual_returns_a = (final_values_a / initial_investment) - 1

print(f"Mean: ${final_values_a.mean():,.2f} ({annual_returns_a.mean()*100:.2f}%)")

# ============================================================================
# STRATEGY B: 100% COP
# ============================================================================
print("Strategy B: All COP (GARCH simulated)...")
final_values_b = []

for sim in range(n_simulations):
    rng = np.random.RandomState(seed=sim)
    sampled_cop = garch_simulate(mu_cop, sigma2_cop_init, omega_cop, alpha_cop, beta_cop, n_weeks, rng)
    final_value = initial_investment * np.prod(1 + sampled_cop)
    final_values_b.append(final_value)

final_values_b = np.array(final_values_b)
annual_returns_b = (final_values_b / initial_investment) - 1

print(f"Mean: ${final_values_b.mean():,.2f} ({annual_returns_b.mean()*100:.2f}%)")

# ============================================================================
# STRATEGY C: 50/50
# ============================================================================
print("Strategy C: 50% COP / 50% S&P 500 (GARCH simulated)...")
final_values_c = []

for sim in range(n_simulations):
    rng = np.random.RandomState(seed=sim)
    sampled_cop = garch_simulate(mu_cop, sigma2_cop_init, omega_cop, alpha_cop, beta_cop, n_weeks, rng)
    sampled_spx = garch_simulate(mu_spx, sigma2_spx_init, omega_spx, alpha_spx, beta_spx, n_weeks, rng)
    
    portfolio_returns = 0.5 * sampled_cop + 0.5 * sampled_spx
    final_value = initial_investment * np.prod(1 + portfolio_returns)
    final_values_c.append(final_value)

final_values_c = np.array(final_values_c)
annual_returns_c = (final_values_c / initial_investment) - 1

print(f"Mean: ${final_values_c.mean():,.2f} ({annual_returns_c.mean()*100:.2f}%)")

# ============================================================================
# STRATEGY D: Dynamic Rebalancing
# ============================================================================
print("Strategy D: Dynamic Rebalancing (GARCH simulated)...")
final_values_d = []

for sim in range(n_simulations):
    rng = np.random.RandomState(seed=sim)
    sampled_cop_full = garch_simulate(mu_cop, sigma2_cop_init, omega_cop, alpha_cop, beta_cop, n_weeks, rng)
    
    # Use different seed for SPX to avoid perfect correlation
    rng2 = np.random.RandomState(seed=sim+10000)
    sampled_spx_full = garch_simulate(mu_spx, sigma2_spx_init, omega_spx, alpha_spx, beta_spx, n_weeks, rng2)
    
    # Initialize portfolio
    value_cop = initial_investment * 0.35
    value_spx = initial_investment * 0.35
    value_cash = initial_investment * 0.30
    
    all_to_cash_next_month = False
    
    # Process each month
    for month_idx, (start_week, end_week) in enumerate(zip(month_starts, month_ends)):
        month_weeks = range(start_week, end_week)
        
        if all_to_cash_next_month:
            for week in month_weeks:
                value_cash *= (1 + cash_weekly_return)
            all_to_cash_next_month = False
            total_value = value_cop + value_spx + value_cash
            value_cop = total_value * 0.35
            value_spx = total_value * 0.35
            value_cash = total_value * 0.30
        else:
            for week in month_weeks:
                value_cop *= (1 + sampled_cop_full[week])
                value_spx *= (1 + sampled_spx_full[week])
                value_cash *= (1 + cash_weekly_return)
            
            cop_month_returns = sampled_cop_full[start_week:end_week]
            spx_month_returns = sampled_spx_full[start_week:end_week]
            
            cop_avg_return = np.mean(cop_month_returns)
            spx_avg_return = np.mean(spx_month_returns)
            
            total_value = value_cop + value_spx + value_cash
            
            if spx_avg_return < 0:
                all_to_cash_next_month = True
                value_cash = total_value
                value_cop = 0
                value_spx = 0
            elif cop_avg_return > spx_avg_return:
                transfer = total_value * 0.10
                value_spx -= transfer
                value_cop += transfer
            else:
                transfer = total_value * 0.10
                value_cop -= transfer
                value_spx += transfer
    
    final_value = value_cop + value_spx + value_cash
    final_values_d.append(final_value)

final_values_d = np.array(final_values_d)
annual_returns_d = (final_values_d / initial_investment) - 1

print(f"Mean: ${final_values_d.mean():,.2f} ({annual_returns_d.mean()*100:.2f}%)")

# ============================================================================
# SAVE RESULTS TO EXCEL
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

with pd.ExcelWriter('Q6_GARCH_Combined_Data.xlsx', engine='openpyxl') as writer:
    # Sheet 1: Wide form returns (4 columns)
    df_wide = pd.DataFrame({
        'Strategy_A_Return': (annual_returns_a * 100).round(2),
        'Strategy_B_Return': (annual_returns_b * 100).round(2),
        'Strategy_C_Return': (annual_returns_c * 100).round(2),
        'Strategy_D_Return': (annual_returns_d * 100).round(2)
    })
    df_wide.to_excel(writer, sheet_name='Returns_Wide_Form', index=False)
    
    # Sheet 2: Wide form complete data
    df_wide_complete = pd.DataFrame({
        'Simulation': range(1, n_simulations + 1),
        'A_Final_Value': final_values_a.round(2),
        'A_Return_%': (annual_returns_a * 100).round(2),
        'B_Final_Value': final_values_b.round(2),
        'B_Return_%': (annual_returns_b * 100).round(2),
        'C_Final_Value': final_values_c.round(2),
        'C_Return_%': (annual_returns_c * 100).round(2),
        'D_Final_Value': final_values_d.round(2),
        'D_Return_%': (annual_returns_d * 100).round(2)
    })
    df_wide_complete.to_excel(writer, sheet_name='All_Data_Wide', index=False)
    
    # Individual strategies
    for label, values in [('A', final_values_a), ('B', final_values_b), 
                          ('C', final_values_c), ('D', final_values_d)]:
        returns = (values / initial_investment) - 1
        df = pd.DataFrame({
            'Simulation': range(1, n_simulations + 1),
            'Final_Value': values.round(2),
            'Annual_Return': returns
        })
        df.to_excel(writer, sheet_name=f'Strategy_{label}', index=False)
    
    # Summary statistics with VaR and Shortfall
    var_5_a = np.percentile(final_values_a, 5)
    var_5_b = np.percentile(final_values_b, 5)
    var_5_c = np.percentile(final_values_c, 5)
    var_5_d = np.percentile(final_values_d, 5)
    
    cvar_5_a = final_values_a[final_values_a <= var_5_a].mean()
    cvar_5_b = final_values_b[final_values_b <= var_5_b].mean()
    cvar_5_c = final_values_c[final_values_c <= var_5_c].mean()
    cvar_5_d = final_values_d[final_values_d <= var_5_d].mean()
    
    stats_data = {
        'Strategy': ['A: All SPX', 'B: All COP', 'C: 50/50', 'D: Rebalance'],
        'Mean_Final_Value': [final_values_a.mean(), final_values_b.mean(), 
                             final_values_c.mean(), final_values_d.mean()],
        'Median_Final_Value': [np.median(final_values_a), np.median(final_values_b),
                               np.median(final_values_c), np.median(final_values_d)],
        'Std_Dev': [final_values_a.std(), final_values_b.std(),
                    final_values_c.std(), final_values_d.std()],
        'Min_Value': [final_values_a.min(), final_values_b.min(),
                      final_values_c.min(), final_values_d.min()],
        'Max_Value': [final_values_a.max(), final_values_b.max(),
                      final_values_c.max(), final_values_d.max()],
        'Mean_Return_%': [annual_returns_a.mean() * 100, annual_returns_b.mean() * 100,
                          annual_returns_c.mean() * 100, annual_returns_d.mean() * 100],
        'P_Loss_%': [(final_values_a < initial_investment).sum() / n_simulations * 100,
                     (final_values_b < initial_investment).sum() / n_simulations * 100,
                     (final_values_c < initial_investment).sum() / n_simulations * 100,
                     (final_values_d < initial_investment).sum() / n_simulations * 100],
        'P_Gain_50_%': [(final_values_a > 150000).sum() / n_simulations * 100,
                        (final_values_b > 150000).sum() / n_simulations * 100,
                        (final_values_c > 150000).sum() / n_simulations * 100,
                        (final_values_d > 150000).sum() / n_simulations * 100],
        'VaR_5%': [var_5_a, var_5_b, var_5_c, var_5_d],
        'Estimated_Shortfall_5%': [cvar_5_a, cvar_5_b, cvar_5_c, cvar_5_d]
    }
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_excel(writer, sheet_name='Summary', index=False)

print("[OK] Excel file created: Q6_GARCH_Combined_Data.xlsx")

# Save individual CSV files
pd.DataFrame({
    'Simulation': range(1, n_simulations + 1),
    'Final_Value': final_values_a,
    'Annual_Return': annual_returns_a
}).to_csv('Strategy_A_GARCH_10000.csv', index=False)

pd.DataFrame({
    'Simulation': range(1, n_simulations + 1),
    'Final_Value': final_values_b,
    'Annual_Return': annual_returns_b
}).to_csv('Strategy_B_GARCH_10000.csv', index=False)

pd.DataFrame({
    'Simulation': range(1, n_simulations + 1),
    'Final_Value': final_values_c,
    'Annual_Return': annual_returns_c
}).to_csv('Strategy_C_GARCH_10000.csv', index=False)

pd.DataFrame({
    'Simulation': range(1, n_simulations + 1),
    'Final_Value': final_values_d,
    'Annual_Return': annual_returns_d
}).to_csv('Strategy_D_GARCH_10000.csv', index=False)

print("[OK] Individual CSV files created")

# ============================================================================
# STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STATISTICS SUMMARY")
print("="*80)
print(df_stats.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Figure 1: Distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('GARCH Distribution of Final Portfolio Values - 10,000 Simulations', 
             fontsize=16, fontweight='bold')

strategies = [
    ('A: All S&P 500', final_values_a, axes[0, 0]),
    ('B: All COP', final_values_b, axes[0, 1]),
    ('C: 50% COP / 50% S&P 500', final_values_c, axes[1, 0]),
    ('D: Dynamic Rebalancing', final_values_d, axes[1, 1])
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for (name, values, ax), color in zip(strategies, colors):
    ax.hist(values, bins=50, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(initial_investment, color='red', linestyle='--', linewidth=2, label='Initial Investment')
    ax.axvline(values.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: ${values.mean():,.0f}')
    ax.set_xlabel('Final Portfolio Value ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Distribution_Comparison_GARCH.png', dpi=300, bbox_inches='tight')
print("[OK] Distribution_Comparison_GARCH.png")
plt.close()

# Figure 2: Box plot
fig, ax = plt.subplots(figsize=(12, 8))

data = [final_values_a, final_values_b, final_values_c, final_values_d]
labels = ['A: All SPX', 'B: All COP', 'C: 50/50', 'D: Rebalance']

bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(initial_investment, color='red', linestyle='--', linewidth=2, label='Initial Investment ($100,000)')
ax.set_ylabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
ax.set_title('GARCH Portfolio Value Distribution by Strategy (Box Plot)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('BoxPlot_Comparison_GARCH.png', dpi=300, bbox_inches='tight')
print("[OK] BoxPlot_Comparison_GARCH.png")
plt.close()

# Figure 3: Risk-Return Profile
fig, ax = plt.subplots(figsize=(12, 8))

means = [annual_returns_a.mean()*100, annual_returns_b.mean()*100, 
         annual_returns_c.mean()*100, annual_returns_d.mean()*100]
stds = [annual_returns_a.std()*100, annual_returns_b.std()*100, 
        annual_returns_c.std()*100, annual_returns_d.std()*100]

scatter = ax.scatter(stds, means, s=500, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

for i, label in enumerate(labels):
    ax.annotate(label, (stds[i], means[i]), fontsize=11, fontweight='bold',
                xytext=(10, 10), textcoords='offset points')

ax.set_xlabel('Risk (Annual Return Std Dev, %)', fontsize=12, fontweight='bold')
ax.set_ylabel('Return (Mean Annual Return, %)', fontsize=12, fontweight='bold')
ax.set_title('GARCH Risk-Return Profile of Strategies', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Risk_Return_Profile_GARCH.png', dpi=300, bbox_inches='tight')
print("[OK] Risk_Return_Profile_GARCH.png")
plt.close()

# Figure 4: Cumulative Distribution
fig, ax = plt.subplots(figsize=(12, 8))

for values, label, color in zip([final_values_a, final_values_b, final_values_c, final_values_d],
                                 labels, colors):
    sorted_vals = np.sort(values)
    cumulative_prob = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cumulative_prob * 100, linewidth=2.5, label=label, color=color)

ax.axvline(initial_investment, color='red', linestyle='--', linewidth=2, label='Initial Investment')
ax.set_xlabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('GARCH Cumulative Distribution of Final Portfolio Values', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Cumulative_Distribution_GARCH.png', dpi=300, bbox_inches='tight')
print("[OK] Cumulative_Distribution_GARCH.png")
plt.close()

# Figure 5: Percentile Comparison
fig, ax = plt.subplots(figsize=(12, 8))

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

for values, label, color in zip([final_values_a, final_values_b, final_values_c, final_values_d],
                                 labels, colors):
    percentile_vals = [np.percentile(values, p) for p in percentiles]
    ax.plot(percentiles, percentile_vals, marker='o', linewidth=2.5, markersize=8, 
            label=label, color=color)

ax.axhline(initial_investment, color='red', linestyle='--', linewidth=2, label='Initial Investment')
ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
ax.set_ylabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
ax.set_title('GARCH Percentile Analysis of Final Portfolio Values', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Percentile_Comparison_GARCH.png', dpi=300, bbox_inches='tight')
print("[OK] Percentile_Comparison_GARCH.png")
plt.close()

print("\n" + "="*80)
print("SIMULATION COMPLETE!")
print("="*80)
print(f"\nAll results saved to Q6_GARCH folder")
print(f"\nOutput files:")
print(f"  Excel: Q6_GARCH_Combined_Data.xlsx")
print(f"  CSV files: Strategy_A/B/C/D_GARCH_10000.csv")
print(f"  Charts: 5 PNG visualization files")
print("="*80 + "\n")

