import pandas as pd

print("\n" + "="*100)
print("SIMPLE SUM ANNUAL RETURN METHOD - DETAILED RESULTS")
print("="*100 + "\n")

# Load the CSV
df = pd.read_csv('bootstrap_annual_returns.csv')

# Display some samples
print("Sample of Bootstrap Annual Returns (First 15 simulations):")
print(df.head(15).to_string(index=False))

print("\n" + "-"*100)
print("\nANNUAL RETURN DISTRIBUTION STATISTICS:")
print("-"*100)

stats = {
    'Count': len(df),
    'Mean': df['Annual_Return'].mean(),
    'Median': df['Annual_Return'].median(),
    'Std Dev': df['Annual_Return'].std(),
    'Min': df['Annual_Return'].min(),
    'Max': df['Annual_Return'].max(),
    'Skewness': df['Annual_Return'].skew(),
    'Kurtosis': df['Annual_Return'].kurtosis()
}

for key, value in stats.items():
    if key in ['Mean', 'Median', 'Std Dev', 'Min', 'Max']:
        print(f"{key:20s}: {value:10.6f} ({value*100:8.2f}%)")
    else:
        print(f"{key:20s}: {value:10.6f}")

print("\n" + "-"*100)
print("\nPROBABILITY ANALYSIS (SIMPLE SUM METHOD):")
print("-"*100)

count_exceed_100 = (df['Annual_Return'] > 1.0).sum()
count_less_0 = (df['Annual_Return'] < 0.0).sum()
count_less_50 = (df['Annual_Return'] < 0.5).sum()

prob_exceed_100 = count_exceed_100 / len(df) * 100
prob_less_0 = count_less_0 / len(df) * 100
prob_less_50 = count_less_50 / len(df) * 100

print(f"\nP(Annual Return > 100%):  {prob_exceed_100:6.2f}%  (Count: {count_exceed_100:,} out of {len(df):,})")
print(f"P(Annual Return < 0%):    {prob_less_0:6.2f}%  (Count: {count_less_0:,} out of {len(df):,})")
print(f"P(Annual Return < 50%):   {prob_less_50:6.2f}%  (Count: {count_less_50:,} out of {len(df):,})")

print("\n" + "="*100)
print("COMPARISON: GEOMETRIC vs SIMPLE SUM")
print("="*100)

comparison_data = {
    'Metric': [
        'P(Return > 100%)',
        'P(Return < 0%)',
        'P(Return < 50%)',
        'Mean Return',
        'Median Return',
        'Std Deviation',
        'Min Return',
        'Max Return'
    ],
    'GEOMETRIC': [
        '9.15%',
        '27.26%',
        '70.32%',
        '31.91%',
        '23.75%',
        '48.57%',
        '-72.30%',
        '327.16%'
    ],
    'SIMPLE_SUM': [
        f'{prob_exceed_100:.2f}%',
        f'{prob_less_0:.2f}%',
        f'{prob_less_50:.2f}%',
        f'{df["Annual_Return"].mean()*100:.2f}%',
        f'{df["Annual_Return"].median()*100:.2f}%',
        f'{df["Annual_Return"].std()*100:.2f}%',
        f'{df["Annual_Return"].min()*100:.2f}%',
        f'{df["Annual_Return"].max()*100:.2f}%'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*100 + "\n")



