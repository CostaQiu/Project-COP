import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../Q52.csv', skiprows=7)
cop_returns = df['COP'].dropna().values * 100  # Convert to percentage
spx_returns = df['SPX'].dropna().values * 100  # Convert to percentage

print("="*80)
print("HISTORICAL WEEKLY RETURNS - DATA VERIFICATION")
print("="*80)
print(f"\nCOP Returns:")
print(f"  Count: {len(cop_returns)}")
print(f"  Mean: {cop_returns.mean():.4f}%")
print(f"  Median: {np.median(cop_returns):.4f}%")
print(f"  Std Dev: {cop_returns.std():.4f}%")
print(f"  Min: {cop_returns.min():.4f}%")
print(f"  Max: {cop_returns.max():.4f}%")

print(f"\nS&P 500 Returns:")
print(f"  Count: {len(spx_returns)}")
print(f"  Mean: {spx_returns.mean():.4f}%")
print(f"  Median: {np.median(spx_returns):.4f}%")
print(f"  Std Dev: {spx_returns.std():.4f}%")
print(f"  Min: {spx_returns.min():.4f}%")
print(f"  Max: {spx_returns.max():.4f}%")

print(f"\nComparison:")
print(f"  COP mean > SPX mean: {cop_returns.mean() > spx_returns.mean()}")
print(f"  Difference: {(cop_returns.mean() - spx_returns.mean()):.4f}%")

# Create histograms with 0.5% bin width, but group tails
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

def create_grouped_bins(returns):
    """Create bins with 0.5% width, grouping <-5% and >5%"""
    # Create 0.5% bins from -5% to 5%
    bins = np.arange(-5.5, 5.5, 0.5)
    
    # Group values < -5%
    below_5_count = np.sum(returns < -5)
    above_5_count = np.sum(returns > 5)
    
    # Get values in range [-5, 5]
    returns_in_range = returns[(returns >= -5) & (returns <= 5)]
    
    return bins, below_5_count, above_5_count, returns_in_range

# Process COP
bins_cop, below_cop, above_cop, cop_in_range = create_grouped_bins(cop_returns)

# Plot COP
ax1 = axes[0]
n, bins_edges, patches = ax1.hist(cop_in_range, bins=bins_cop, alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=1.5)

# Add grouped tail bars
if below_cop > 0:
    ax1.bar(-5.25, below_cop, width=0.4, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=1.5)
if above_cop > 0:
    ax1.bar(5.25, above_cop, width=0.4, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add text labels for grouped bins
if below_cop > 0:
    ax1.text(-5.25, below_cop + 1, f'n={int(below_cop)}\n<-5%', ha='center', fontsize=9, fontweight='bold')
if above_cop > 0:
    ax1.text(5.25, above_cop + 1, f'n={int(above_cop)}\n>5%', ha='center', fontsize=9, fontweight='bold')

ax1.axvline(cop_returns.mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {cop_returns.mean():.2f}%')
ax1.axvline(np.median(cop_returns), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(cop_returns):.2f}%')
ax1.set_xlabel('Weekly Return (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('COP Historical Weekly Returns (0.5% bins)\n260 weeks of data', fontsize=13, fontweight='bold')
ax1.set_xlim(-6, 6)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Process SPX
bins_spx, below_spx, above_spx, spx_in_range = create_grouped_bins(spx_returns)

# Plot SPX
ax2 = axes[1]
n, bins_edges, patches = ax2.hist(spx_in_range, bins=bins_spx, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=1.5)

# Add grouped tail bars
if below_spx > 0:
    ax2.bar(-5.25, below_spx, width=0.4, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.5)
if above_spx > 0:
    ax2.bar(5.25, above_spx, width=0.4, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add text labels for grouped bins
if below_spx > 0:
    ax2.text(-5.25, below_spx + 1, f'n={int(below_spx)}\n<-5%', ha='center', fontsize=9, fontweight='bold')
if above_spx > 0:
    ax2.text(5.25, above_spx + 1, f'n={int(above_spx)}\n>5%', ha='center', fontsize=9, fontweight='bold')

ax2.axvline(spx_returns.mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {spx_returns.mean():.2f}%')
ax2.axvline(np.median(spx_returns), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(spx_returns):.2f}%')
ax2.set_xlabel('Weekly Return (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('S&P 500 Historical Weekly Returns (0.5% bins)\n260 weeks of data', fontsize=13, fontweight='bold')
ax2.set_xlim(-6, 6)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle('Historical Weekly Returns Distribution (0.5% bins, grouped tails)', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('Historical_Returns_Distribution.png', dpi=300, bbox_inches='tight')
print(f"\n[OK] Saved: Historical_Returns_Distribution.png")
plt.close()

print("\nClear conclusion: COP has HIGHER mean return than SPX historically!")
print(f"GARCH must preserve this relationship.")

