import pandas as pd
import numpy as np

# Read the CSV file, skipping metadata rows
df = pd.read_csv('Q52.csv', skiprows=7)

# Display first few rows to verify data structure
print("Data structure verification:")
print(df.head(10))
print(f"\nColumn names: {df.columns.tolist()}")
print(f"Data shape: {df.shape}")

# Extract COP returns (skip NaN values)
cop_returns = df['COP'].dropna().values

print(f"\n{'='*60}")
print(f"DATA SUMMARY - COP Weekly Returns")
print(f"{'='*60}")
print(f"Number of weekly returns available: {len(cop_returns)}")
print(f"Mean weekly return: {cop_returns.mean():.6f} ({cop_returns.mean()*100:.4f}%)")
print(f"Std Dev weekly return: {cop_returns.std():.6f} ({cop_returns.std()*100:.4f}%)")
print(f"Min weekly return: {cop_returns.min():.6f} ({cop_returns.min()*100:.4f}%)")
print(f"Max weekly return: {cop_returns.max():.6f} ({cop_returns.max()*100:.4f}%)")

# Set random seed for reproducibility
np.random.seed(42)

# Run Monte Carlo simulation
n_simulations = 10000
n_weeks = 52
annual_returns = []

print(f"\n{'='*60}")
print(f"Running Monte Carlo Simulation with {n_simulations:,} iterations...")
print(f"{'='*60}")

for i in range(n_simulations):
    # Randomly select 52 weekly returns with replacement (bootstrap)
    sampled_returns = np.random.choice(cop_returns, size=n_weeks, replace=True)
    
    # Calculate annual return GEOMETRICALLY (Original method - ACTIVE)
    # Annual Return = (1 + r1) * (1 + r2) * ... * (1 + r52) - 1
    annual_return = np.prod(1 + sampled_returns) - 1
    
    # Calculate annual return as SIMPLE SUM of weekly returns (Alternative method - DISABLED)
    # Annual Return = r1 + r2 + ... + r52
    # annual_return = np.sum(sampled_returns)
    
    annual_returns.append(annual_return)

annual_returns = np.array(annual_returns)

# Calculate probabilities
prob_exceed_100 = np.sum(annual_returns > 1.0) / n_simulations * 100
prob_less_0 = np.sum(annual_returns < 0.0) / n_simulations * 100
prob_less_50 = np.sum(annual_returns < 0.5) / n_simulations * 100

# Display results
print(f"\n{'='*60}")
print(f"MONTE CARLO SIMULATION RESULTS")
print(f"{'='*60}")
print(f"\nAnnual Returns Statistics (from {n_simulations:,} simulations):")
print(f"  Mean Annual Return:    {annual_returns.mean():.6f} ({annual_returns.mean()*100:7.2f}%)")
print(f"  Median Annual Return:  {np.median(annual_returns):.6f} ({np.median(annual_returns)*100:7.2f}%)")
print(f"  Std Dev:               {annual_returns.std():.6f} ({annual_returns.std()*100:7.2f}%)")
print(f"  Min Annual Return:     {annual_returns.min():.6f} ({annual_returns.min()*100:7.2f}%)")
print(f"  Max Annual Return:     {annual_returns.max():.6f} ({annual_returns.max()*100:7.2f}%)")

print(f"\n{'='*60}")
print(f"PROBABILITY ANALYSIS")
print(f"{'='*60}")
print(f"  P(Annual Return > 100%):  {prob_exceed_100:6.2f}%  (Count: {np.sum(annual_returns > 1.0):,})")
print(f"  P(Annual Return < 0%):    {prob_less_0:6.2f}%  (Count: {np.sum(annual_returns < 0.0):,})")
print(f"  P(Annual Return < 50%):   {prob_less_50:6.2f}%  (Count: {np.sum(annual_returns < 0.5):,})")
print(f"{'='*60}\n")

# Additional percentiles for insight
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"Percentile Distribution of Annual Returns:")
for p in percentiles:
    val = np.percentile(annual_returns, p)
    print(f"  {p}th percentile: {val:.6f} ({val*100:7.2f}%)")

# Save to CSV
results_df = pd.DataFrame({
    'Simulation_Number': np.arange(1, n_simulations + 1),
    'Annual_Return': annual_returns
})

output_path = 'bootstrap_annual_returns.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"  File contains {len(results_df):,} bootstrap annual returns")

