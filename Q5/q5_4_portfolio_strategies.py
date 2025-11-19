import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create output directory
output_dir = Path('Q5.4')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("QUESTION 5.4: PORTFOLIO STRATEGY COMPARISON - MONTE CARLO SIMULATION")
print("="*80)

# Load data
df = pd.read_csv('Q52.csv', skiprows=7)
cop_returns = df['COP'].dropna().values
spx_returns = df['SPX'].dropna().values

print(f"\nData loaded:")
print(f"  COP weekly returns: {len(cop_returns)} weeks")
print(f"  S&P 500 weekly returns: {len(spx_returns)} weeks")
print(f"  Mean COP weekly return: {cop_returns.mean()*100:.4f}%")
print(f"  Mean S&P 500 weekly return: {spx_returns.mean()*100:.4f}%")

# Simulation parameters
n_simulations = 10000
n_weeks = 52
initial_investment = 100000
cash_weekly_return = 0.01 / 52  # 1% annual = 0.01/52 weekly

# Define month structure (4-5-4-5-4-4-5-4-4-5-4-4 pattern for 52 weeks)
# Month end weeks: 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48, 52
month_ends = [5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48, 52]
month_starts = [0] + month_ends[:-1]

print(f"\nSimulation parameters:")
print(f"  Number of simulations: {n_simulations:,}")
print(f"  Investment period: {n_weeks} weeks")
print(f"  Rebalancing periods: {len(month_ends)} months")
print(f"  Initial investment: ${initial_investment:,.2f}")
print(f"  Month structure: {month_ends}")

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STRATEGY A: Invest all in S&P 500 (Buy & Hold)
# ============================================================================
print("\n" + "="*80)
print("STRATEGY A: All S&P 500 (Buy & Hold)")
print("="*80)

final_values_a = []

for sim in range(n_simulations):
    # Bootstrap sample 52 weeks
    sampled_spx = np.random.choice(spx_returns, size=n_weeks, replace=True)
    
    # Calculate final value (geometric return)
    final_value = initial_investment * np.prod(1 + sampled_spx)
    final_values_a.append(final_value)

final_values_a = np.array(final_values_a)
annual_returns_a = (final_values_a / initial_investment) - 1

print(f"Mean final value: ${final_values_a.mean():,.2f}")
print(f"Median final value: ${np.median(final_values_a):,.2f}")
print(f"Mean annual return: {annual_returns_a.mean()*100:.2f}%")
print(f"Min final value: ${final_values_a.min():,.2f}")
print(f"Max final value: ${final_values_a.max():,.2f}")

# ============================================================================
# STRATEGY B: Invest all in COP (Buy & Hold)
# ============================================================================
print("\n" + "="*80)
print("STRATEGY B: All COP (Buy & Hold)")
print("="*80)

final_values_b = []

for sim in range(n_simulations):
    # Bootstrap sample 52 weeks
    sampled_cop = np.random.choice(cop_returns, size=n_weeks, replace=True)
    
    # Calculate final value (geometric return)
    final_value = initial_investment * np.prod(1 + sampled_cop)
    final_values_b.append(final_value)

final_values_b = np.array(final_values_b)
annual_returns_b = (final_values_b / initial_investment) - 1

print(f"Mean final value: ${final_values_b.mean():,.2f}")
print(f"Median final value: ${np.median(final_values_b):,.2f}")
print(f"Mean annual return: {annual_returns_b.mean()*100:.2f}%")
print(f"Min final value: ${final_values_b.min():,.2f}")
print(f"Max final value: ${final_values_b.max():,.2f}")

# ============================================================================
# STRATEGY C: 50/50 Portfolio (Buy & Hold)
# ============================================================================
print("\n" + "="*80)
print("STRATEGY C: 50% COP / 50% S&P 500 (Buy & Hold)")
print("="*80)

final_values_c = []

for sim in range(n_simulations):
    # Bootstrap sample 52 weeks
    sampled_cop = np.random.choice(cop_returns, size=n_weeks, replace=True)
    sampled_spx = np.random.choice(spx_returns, size=n_weeks, replace=True)
    
    # Calculate 50/50 portfolio return
    portfolio_returns = 0.5 * sampled_cop + 0.5 * sampled_spx
    
    # Calculate final value (geometric return)
    final_value = initial_investment * np.prod(1 + portfolio_returns)
    final_values_c.append(final_value)

final_values_c = np.array(final_values_c)
annual_returns_c = (final_values_c / initial_investment) - 1

print(f"Mean final value: ${final_values_c.mean():,.2f}")
print(f"Median final value: ${np.median(final_values_c):,.2f}")
print(f"Mean annual return: {annual_returns_c.mean()*100:.2f}%")
print(f"Min final value: ${final_values_c.min():,.2f}")
print(f"Max final value: ${final_values_c.max():,.2f}")

# ============================================================================
# STRATEGY D: Dynamic Rebalancing (35% COP / 35% SPX / 30% Cash + Monthly Rebalance)
# ============================================================================
print("\n" + "="*80)
print("STRATEGY D: Dynamic Rebalancing (35% COP / 35% SPX / 30% Cash)")
print("="*80)

final_values_d = []

for sim in range(n_simulations):
    # Bootstrap sample 52 weeks (we'll use same weeks but resample returns)
    week_indices = np.random.choice(len(cop_returns), size=n_weeks, replace=True)
    sampled_cop_full = cop_returns[week_indices]
    sampled_spx_full = spx_returns[week_indices]
    
    # Initialize portfolio
    value_cop = initial_investment * 0.35
    value_spx = initial_investment * 0.35
    value_cash = initial_investment * 0.30
    
    # Track for rebalancing (all to cash flag)
    all_to_cash_next_month = False
    
    # Process each month
    for month_idx, (start_week, end_week) in enumerate(zip(month_starts, month_ends)):
        # Get weeks for this month
        month_weeks = range(start_week, end_week)
        
        # If previous month was all cash, keep all in cash for this month
        if all_to_cash_next_month:
            # All stays in cash (earning 1% annual return = 0.01/52 weekly)
            for week in month_weeks:
                value_cash *= (1 + cash_weekly_return)
            all_to_cash_next_month = False
            # Reset to 35/35/30
            total_value = value_cop + value_spx + value_cash
            value_cop = total_value * 0.35
            value_spx = total_value * 0.35
            value_cash = total_value * 0.30
        else:
            # Process weeks and accumulate returns
            for week in month_weeks:
                value_cop *= (1 + sampled_cop_full[week])
                value_spx *= (1 + sampled_spx_full[week])
                value_cash *= (1 + cash_weekly_return)
            
            # At end of month, calculate average returns for last 4-5 weeks
            n_weeks_in_month = end_week - start_week
            cop_month_returns = sampled_cop_full[start_week:end_week]
            spx_month_returns = sampled_spx_full[start_week:end_week]
            
            cop_avg_return = np.mean(cop_month_returns)
            spx_avg_return = np.mean(spx_month_returns)
            
            # Apply rebalancing rules
            total_value = value_cop + value_spx + value_cash
            
            if spx_avg_return < 0:
                # Market had negative return: move 100% to cash for entire next month
                all_to_cash_next_month = True
                value_cash = total_value
                value_cop = 0
                value_spx = 0
            elif cop_avg_return > spx_avg_return:
                # COP performed better: move 10% from Market to Stock
                transfer = total_value * 0.10
                value_spx -= transfer
                value_cop += transfer
            else:
                # Market performed better or equal: move 10% from Stock to Market
                transfer = total_value * 0.10
                value_cop -= transfer
                value_spx += transfer
    
    # Calculate final value
    final_value = value_cop + value_spx + value_cash
    final_values_d.append(final_value)

final_values_d = np.array(final_values_d)
annual_returns_d = (final_values_d / initial_investment) - 1

print(f"Mean final value: ${final_values_d.mean():,.2f}")
print(f"Median final value: ${np.median(final_values_d):,.2f}")
print(f"Mean annual return: {annual_returns_d.mean()*100:.2f}%")
print(f"Min final value: ${final_values_d.min():,.2f}")
print(f"Max final value: ${final_values_d.max():,.2f}")

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS TO CSV FILES")
print("="*80)

# Strategy A
df_a = pd.DataFrame({
    'Simulation': np.arange(1, n_simulations + 1),
    'Final_Value': final_values_a,
    'Annual_Return': annual_returns_a
})
df_a.to_csv(output_dir / 'Strategy_A_SPX_10000.csv', index=False)
print(f"Strategy A saved: {output_dir / 'Strategy_A_SPX_10000.csv'}")

# Strategy B
df_b = pd.DataFrame({
    'Simulation': np.arange(1, n_simulations + 1),
    'Final_Value': final_values_b,
    'Annual_Return': annual_returns_b
})
df_b.to_csv(output_dir / 'Strategy_B_COP_10000.csv', index=False)
print(f"Strategy B saved: {output_dir / 'Strategy_B_COP_10000.csv'}")

# Strategy C
df_c = pd.DataFrame({
    'Simulation': np.arange(1, n_simulations + 1),
    'Final_Value': final_values_c,
    'Annual_Return': annual_returns_c
})
df_c.to_csv(output_dir / 'Strategy_C_5050_10000.csv', index=False)
print(f"Strategy C saved: {output_dir / 'Strategy_C_5050_10000.csv'}")

# Strategy D
df_d = pd.DataFrame({
    'Simulation': np.arange(1, n_simulations + 1),
    'Final_Value': final_values_d,
    'Annual_Return': annual_returns_d
})
df_d.to_csv(output_dir / 'Strategy_D_Rebalance_10000.csv', index=False)
print(f"Strategy D saved: {output_dir / 'Strategy_D_Rebalance_10000.csv'}")

# ============================================================================
# COMPREHENSIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE STATISTICS COMPARISON")
print("="*80)

stats_data = {
    'Strategy': ['A: All SPX', 'B: All COP', 'C: 50/50', 'D: Rebalance'],
    'Mean Final Value': [
        f"${final_values_a.mean():,.2f}",
        f"${final_values_b.mean():,.2f}",
        f"${final_values_c.mean():,.2f}",
        f"${final_values_d.mean():,.2f}"
    ],
    'Median Final Value': [
        f"${np.median(final_values_a):,.2f}",
        f"${np.median(final_values_b):,.2f}",
        f"${np.median(final_values_c):,.2f}",
        f"${np.median(final_values_d):,.2f}"
    ],
    'Std Dev': [
        f"${final_values_a.std():,.2f}",
        f"${final_values_b.std():,.2f}",
        f"${final_values_c.std():,.2f}",
        f"${final_values_d.std():,.2f}"
    ],
    'Min Value': [
        f"${final_values_a.min():,.2f}",
        f"${final_values_b.min():,.2f}",
        f"${final_values_c.min():,.2f}",
        f"${final_values_d.min():,.2f}"
    ],
    'Max Value': [
        f"${final_values_a.max():,.2f}",
        f"${final_values_b.max():,.2f}",
        f"${final_values_c.max():,.2f}",
        f"${final_values_d.max():,.2f}"
    ],
    'Mean Return %': [
        f"{annual_returns_a.mean()*100:.2f}%",
        f"{annual_returns_b.mean()*100:.2f}%",
        f"{annual_returns_c.mean()*100:.2f}%",
        f"{annual_returns_d.mean()*100:.2f}%"
    ],
    'P(Loss)': [
        f"{np.sum(final_values_a < initial_investment)/n_simulations*100:.2f}%",
        f"{np.sum(final_values_b < initial_investment)/n_simulations*100:.2f}%",
        f"{np.sum(final_values_c < initial_investment)/n_simulations*100:.2f}%",
        f"{np.sum(final_values_d < initial_investment)/n_simulations*100:.2f}%"
    ],
    'P(Gain >50%)': [
        f"{np.sum((final_values_a - initial_investment) / initial_investment > 0.5)/n_simulations*100:.2f}%",
        f"{np.sum((final_values_b - initial_investment) / initial_investment > 0.5)/n_simulations*100:.2f}%",
        f"{np.sum((final_values_c - initial_investment) / initial_investment > 0.5)/n_simulations*100:.2f}%",
        f"{np.sum((final_values_d - initial_investment) / initial_investment > 0.5)/n_simulations*100:.2f}%"
    ]
}

stats_df = pd.DataFrame(stats_data)
print("\n" + stats_df.to_string(index=False))

# Save statistics
stats_df.to_csv(output_dir / 'Statistics_Summary.csv', index=False)
print(f"\nStatistics saved: {output_dir / 'Statistics_Summary.csv'}")

# ============================================================================
# ADVANCED STATISTICS
# ============================================================================
print("\n" + "="*80)
print("ADVANCED STATISTICS")
print("="*80)

advanced_stats = {
    'Strategy': ['A: All SPX', 'B: All COP', 'C: 50/50', 'D: Rebalance'],
    'Skewness': [
        f"{pd.Series(final_values_a).skew():.4f}",
        f"{pd.Series(final_values_b).skew():.4f}",
        f"{pd.Series(final_values_c).skew():.4f}",
        f"{pd.Series(final_values_d).skew():.4f}"
    ],
    'Kurtosis': [
        f"{pd.Series(final_values_a).kurtosis():.4f}",
        f"{pd.Series(final_values_b).kurtosis():.4f}",
        f"{pd.Series(final_values_c).kurtosis():.4f}",
        f"{pd.Series(final_values_d).kurtosis():.4f}"
    ],
    'Sharpe Ratio (RF=1%)': [
        f"{(annual_returns_a.mean() - 0.01) / annual_returns_a.std():.4f}",
        f"{(annual_returns_b.mean() - 0.01) / annual_returns_b.std():.4f}",
        f"{(annual_returns_c.mean() - 0.01) / annual_returns_c.std():.4f}",
        f"{(annual_returns_d.mean() - 0.01) / annual_returns_d.std():.4f}"
    ],
    'VaR 5%': [
        f"${np.percentile(final_values_a, 5):,.2f}",
        f"${np.percentile(final_values_b, 5):,.2f}",
        f"${np.percentile(final_values_c, 5):,.2f}",
        f"${np.percentile(final_values_d, 5):,.2f}"
    ],
    'CVaR 5%': [
        f"${final_values_a[final_values_a <= np.percentile(final_values_a, 5)].mean():,.2f}",
        f"${final_values_b[final_values_b <= np.percentile(final_values_b, 5)].mean():,.2f}",
        f"${final_values_c[final_values_c <= np.percentile(final_values_c, 5)].mean():,.2f}",
        f"${final_values_d[final_values_d <= np.percentile(final_values_d, 5)].mean():,.2f}"
    ]
}

advanced_stats_df = pd.DataFrame(advanced_stats)
print("\n" + advanced_stats_df.to_string(index=False))

advanced_stats_df.to_csv(output_dir / 'Advanced_Statistics.csv', index=False)
print(f"\nAdvanced statistics saved: {output_dir / 'Advanced_Statistics.csv'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Figure 1: Distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution of Final Portfolio Values - 10,000 Simulations', fontsize=16, fontweight='bold')

strategies = [
    ('A: All S&P 500', final_values_a, axes[0, 0]),
    ('B: All COP', final_values_b, axes[0, 1]),
    ('C: 50% COP / 50% S&P 500', final_values_c, axes[1, 0]),
    ('D: Dynamic Rebalancing', final_values_d, axes[1, 1])
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for name, values, ax, color in zip([s[0] for s in strategies], [s[1] for s in strategies], 
                                     [s[2] for s in strategies], colors):
    ax.hist(values, bins=50, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(initial_investment, color='red', linestyle='--', linewidth=2, label='Initial Investment')
    ax.axvline(values.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: ${values.mean():,.0f}')
    ax.set_xlabel('Final Portfolio Value ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'Distribution_Comparison.png', dpi=300, bbox_inches='tight')
print(f"Chart 1 saved: {output_dir / 'Distribution_Comparison.png'}")
plt.close()

# Figure 2: Box plot comparison
fig, ax = plt.subplots(figsize=(12, 8))

data_for_box = [final_values_a, final_values_b, final_values_c, final_values_d]
labels = ['A: All SPX', 'B: All COP', 'C: 50/50', 'D: Rebalance']

bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True, notch=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(initial_investment, color='red', linestyle='--', linewidth=2, label='Initial Investment ($100,000)')
ax.set_ylabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
ax.set_title('Portfolio Value Distribution by Strategy (Box Plot)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'BoxPlot_Comparison.png', dpi=300, bbox_inches='tight')
print(f"Chart 2 saved: {output_dir / 'BoxPlot_Comparison.png'}")
plt.close()

# Figure 3: Mean vs Risk (Return vs Volatility)
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
ax.set_title('Risk-Return Profile of Strategies', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'Risk_Return_Profile.png', dpi=300, bbox_inches='tight')
print(f"Chart 3 saved: {output_dir / 'Risk_Return_Profile.png'}")
plt.close()

# Figure 4: Cumulative probability
fig, ax = plt.subplots(figsize=(12, 8))

for values, label, color in zip([final_values_a, final_values_b, final_values_c, final_values_d],
                                 labels, colors):
    sorted_vals = np.sort(values)
    cumulative_prob = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cumulative_prob * 100, linewidth=2.5, label=label, color=color)

ax.axvline(initial_investment, color='red', linestyle='--', linewidth=2, label='Initial Investment')
ax.set_xlabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Distribution of Final Portfolio Values', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'Cumulative_Distribution.png', dpi=300, bbox_inches='tight')
print(f"Chart 4 saved: {output_dir / 'Cumulative_Distribution.png'}")
plt.close()

# Figure 5: Percentile comparison
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
ax.set_title('Percentile Analysis of Final Portfolio Values', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'Percentile_Comparison.png', dpi=300, bbox_inches='tight')
print(f"Chart 5 saved: {output_dir / 'Percentile_Comparison.png'}")
plt.close()

print("\n" + "="*80)
print("SIMULATION COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {output_dir}")
print(f"\nOutput files:")
print(f"  - Strategy_A_SPX_10000.csv")
print(f"  - Strategy_B_COP_10000.csv")
print(f"  - Strategy_C_5050_10000.csv")
print(f"  - Strategy_D_Rebalance_10000.csv")
print(f"  - Statistics_Summary.csv")
print(f"  - Advanced_Statistics.csv")
print(f"  - Distribution_Comparison.png")
print(f"  - BoxPlot_Comparison.png")
print(f"  - Risk_Return_Profile.png")
print(f"  - Cumulative_Distribution.png")
print(f"  - Percentile_Comparison.png")
print("="*80 + "\n")




