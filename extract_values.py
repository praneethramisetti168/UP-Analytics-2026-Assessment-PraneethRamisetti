import pandas as pd
import numpy as np

test = pd.read_csv('aps_failure_test_set.csv', na_values=['na'], skiprows=20)
pred = pd.read_csv('outputs/test_predictions.csv')

fields = ['aa_000','ab_000','cr_000','cs_000','ee_000','ag_004','ay_006','ay_000','ba_000','bj_000']

print("=== HIGH RISK (actual APS failure) ===")
high_idx = pred[(pred['risk_bucket']=='High') & (pred['actual_class']==1)].index
row = test.iloc[high_idx[0]]
for f in fields:
    val = row[f] if f in row.index else 'N/A'
    print(f"  {f}: {val}")

print()
print("=== MEDIUM RISK (actual APS failure) ===")
med_idx = pred[(pred['risk_bucket']=='Medium') & (pred['actual_class']==1)].index
row2 = test.iloc[med_idx[0]]
for f in fields:
    val = row2[f] if f in row2.index else 'N/A'
    print(f"  {f}: {val}")

print()
print("=== LOW RISK (safe truck) ===")
low_idx = pred[(pred['risk_bucket']=='Low') & (pred['actual_class']==0)].index
row3 = test.iloc[low_idx[0]]
for f in fields:
    val = row3[f] if f in row3.index else 'N/A'
    print(f"  {f}: {val}")
