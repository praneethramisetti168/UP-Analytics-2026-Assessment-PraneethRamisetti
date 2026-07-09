"""
APS Failure at Scania Trucks — Full Analysis Script
Run this to reproduce all results and generate model artifacts + predictions.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import shap

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE = Path(r"c:\Users\prane\Downloads\UP-Analytics-2026-Assessment-PraneethRamisetti")
TRAIN_PATH = BASE / "aps_failure_training_set.csv"
TEST_PATH  = BASE / "aps_failure_test_set.csv"
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR  = BASE / "api" / "model_artifacts"
FIG_DIR    = BASE / "outputs" / "figures"

for d in [OUTPUT_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*60)
print("APS FAILURE PREDICTION — SCANIA TRUCKS")
print("="*60)

# ─────────────────────────────────────────────
# ---------------------------------------------
# SECTION 1: Data Loading & Label Encoding
# ---------------------------------------------
print("\n[1] Loading data...")
train_raw = pd.read_csv(TRAIN_PATH, na_values=["na"])
test_raw  = pd.read_csv(TEST_PATH,  na_values=["na"])

LABEL_MAP = {"pos": 1, "neg": 0}
print("    Label mapping: pos->1 (APS failure), neg->0 (non-APS failure)")

train_raw["target"] = train_raw["class"].map(LABEL_MAP)
test_raw["target"]  = test_raw["class"].map(LABEL_MAP)

feature_cols = [c for c in train_raw.columns if c not in ["class", "target"]]

X_train_raw = train_raw[feature_cols].copy()
y_train     = train_raw["target"].copy()
X_test_raw  = test_raw[feature_cols].copy()
y_test      = test_raw["target"].copy()

print(f"    Train: {X_train_raw.shape}, positives: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"    Test:  {X_test_raw.shape},  positives: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ─────────────────────────────────────────────
# SECTION 2: EDA
# ─────────────────────────────────────────────
print("\n[2] EDA...")

# 2a. Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes:
    ax.set_facecolor('#1a1a2e')

counts = y_train.value_counts()
colors = ['#00d4ff', '#ff6b6b']
bars = axes[0].bar(['Non-APS (neg)', 'APS Failure (pos)'], [counts[0], counts[1]],
                   color=colors, edgecolor='white', linewidth=0.5, width=0.5)
axes[0].set_title('Class Distribution — Training Set', color='white', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Count', color='white', fontsize=12)
axes[0].tick_params(colors='white')
for spine in axes[0].spines.values():
    spine.set_edgecolor('#333355')
for bar, count in zip(bars, [counts[0], counts[1]]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{count:,}\n({count/len(y_train)*100:.1f}%)',
                 ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

axes[1].pie([counts[0], counts[1]], labels=['Non-APS\n(neg)', 'APS Failure\n(pos)'],
            colors=colors, autopct='%1.2f%%', startangle=90,
            textprops={'color': 'white', 'fontsize': 11},
            wedgeprops={'edgecolor': '#0f0f1a', 'linewidth': 2})
axes[1].set_title('Class Imbalance Ratio', color='white', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 01_class_distribution.png")

# 2b. Top sensor distributions by class
pos_mask = y_train == 1
neg_mask = y_train == 0

# Select top informative sensors by APS-positive mean
means_pos = X_train_raw[pos_mask].mean()
means_neg = X_train_raw[neg_mask].mean()
ratio = (means_pos + 1) / (means_neg + 1)
top10 = ratio.nlargest(10).index.tolist()

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.patch.set_facecolor('#0f0f1a')
axes = axes.flatten()
for i, col in enumerate(top10):
    ax = axes[i]
    ax.set_facecolor('#1a1a2e')
    data_neg = X_train_raw.loc[neg_mask, col].dropna()
    data_pos = X_train_raw.loc[pos_mask, col].dropna()
    ax.hist(data_neg, bins=40, alpha=0.6, color='#00d4ff', label='Non-APS', density=True)
    ax.hist(data_pos, bins=40, alpha=0.6, color='#ff6b6b', label='APS Fail', density=True)
    ax.set_title(col, color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    if i == 0:
        ax.legend(fontsize=8, facecolor='#0f0f1a', labelcolor='white')

fig.suptitle('Top 10 Discriminative Sensors: APS vs Non-APS Distribution',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_top_sensor_distributions.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 02_top_sensor_distributions.png")

# 2c. Feature correlation heatmap (top 20)
top20_cols = ratio.nlargest(20).index.tolist()
corr = X_train_raw[top20_cols].corr()
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#1a1a2e')
sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            annot=False, linewidths=0.3, linecolor='#0f0f1a',
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap (Top 20 Sensors)', color='white',
             fontsize=14, fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=8)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 03_correlation_heatmap.png")

# ─────────────────────────────────────────────
# SECTION 3: Missing Data Assessment
# ─────────────────────────────────────────────
print("\n[3] Missing data assessment...")
miss_pct = (X_train_raw.isnull().sum() / len(X_train_raw) * 100).sort_values(ascending=False)
miss_pct_nonzero = miss_pct[miss_pct > 0]

print(f"    Features with any missing: {len(miss_pct_nonzero)}")
print(f"    Max missing: {miss_pct_nonzero.max():.1f}%  |  Median: {miss_pct_nonzero.median():.1f}%")

# Plot missingness
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#0f0f1a')

# Bar chart of top-30 most missing
ax = axes[0]
ax.set_facecolor('#1a1a2e')
top30 = miss_pct_nonzero.head(30)
colors_bar = ['#ff4444' if v > 70 else '#ffaa00' if v > 40 else '#00d4ff' for v in top30.values]
bars = ax.barh(range(len(top30)), top30.values, color=colors_bar, edgecolor='white', linewidth=0.3)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30.index, fontsize=8, color='white')
ax.set_xlabel('Missing %', color='white', fontsize=11)
ax.set_title('Top 30 Features by Missingness', color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')
ax.axvline(70, color='#ff4444', linestyle='--', alpha=0.7, label='70% drop threshold')
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

# Histogram of missingness distribution
ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')
ax2.hist(miss_pct_nonzero.values, bins=30, color='#7c3aed', edgecolor='white', linewidth=0.5, alpha=0.85)
ax2.set_xlabel('Missing %', color='white', fontsize=11)
ax2.set_ylabel('Number of Features', color='white', fontsize=11)
ax2.set_title('Distribution of Missingness Across Features', color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#333355')

# Annotations
drop_count = (miss_pct > 70).sum()
impute_count = ((miss_pct > 0) & (miss_pct <= 70)).sum()
ax2.text(0.98, 0.95, f'Drop (>70%): {drop_count} cols\nImpute (≤70%): {impute_count} cols\nComplete: {(miss_pct==0).sum()} cols',
         transform=ax2.transAxes, ha='right', va='top',
         color='white', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#2a2a4a', alpha=0.8))

plt.tight_layout()
plt.savefig(FIG_DIR / "04_missing_data.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 04_missing_data.png")

# ─────────────────────────────────────────────
# SECTION 4 & 5: Feature Engineering + Pipeline
# ─────────────────────────────────────────────
print("\n[4] Feature engineering & preprocessing pipeline...")

# Identify columns to drop (>70% missing in train)
drop_cols = miss_pct[miss_pct > 70].index.tolist()
keep_cols = [c for c in feature_cols if c not in drop_cols]
print(f"    Dropping {len(drop_cols)} high-missing columns")
print(f"    Keeping {len(keep_cols)} feature columns")

# Missingness indicator flags for cols with 20-70% missing
high_miss_cols = miss_pct[(miss_pct >= 20) & (miss_pct <= 70)].index.tolist()
print(f"    Adding {len(high_miss_cols)} missingness indicator flags")

def add_miss_flags(X, cols):
    Xc = X.copy()
    for c in cols:
        if c in Xc.columns:
            Xc[f"{c}_missing"] = Xc[c].isnull().astype(int)
    return Xc

X_train_fe = add_miss_flags(X_train_raw[keep_cols], high_miss_cols)
X_test_fe  = add_miss_flags(X_test_raw[keep_cols], high_miss_cols)

# Row-level summary features
for df in [X_train_fe, X_test_fe]:
    orig_numeric = [c for c in keep_cols if c in df.columns]
    df['row_null_count']  = df[orig_numeric].isnull().sum(axis=1)
    df['row_null_frac']   = df['row_null_count'] / len(orig_numeric)
    df['row_mean']        = df[orig_numeric].mean(axis=1)
    df['row_std']         = df[orig_numeric].std(axis=1)
    df['row_max']         = df[orig_numeric].max(axis=1)
    df['row_min']         = df[orig_numeric].min(axis=1)

all_feature_cols = X_train_fe.columns.tolist()
print(f"    Final feature count: {len(all_feature_cols)}")

# Preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

X_train_proc = preprocessor.fit_transform(X_train_fe)
X_test_proc  = preprocessor.transform(X_test_fe)

# Save preprocessor + feature column list
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
joblib.dump({'feature_cols': keep_cols, 'high_miss_cols': high_miss_cols}, 
            MODEL_DIR / "feature_config.pkl")
print("    Preprocessor saved.")

# ─────────────────────────────────────────────
# SECTION 6: Time-based Train/Val Split
# ─────────────────────────────────────────────
print("\n[5] Time-based train/validation split (positional ordering)...")
n_train = len(X_train_proc)
split_idx = int(n_train * 0.80)

X_tr  = X_train_proc[:split_idx]
y_tr  = y_train.values[:split_idx]
X_val = X_train_proc[split_idx:]
y_val = y_train.values[split_idx:]

print(f"    Train split: {X_tr.shape[0]} rows ({y_tr.sum()} positives)")
print(f"    Val  split:  {X_val.shape[0]} rows ({y_val.sum()} positives)")

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"    scale_pos_weight = {scale_pos_weight:.1f}")

# ─────────────────────────────────────────────
# SECTION 7: Model Training
# ─────────────────────────────────────────────
print("\n[6] Training models...")

results = {}

# ── Model 1: Logistic Regression (baseline)
print("    [1/4] Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.1)
lr.fit(X_tr, y_tr)
lr_probs_val = lr.predict_proba(X_val)[:, 1]
lr_probs_test = lr.predict_proba(X_test_proc)[:, 1]
ap_lr = average_precision_score(y_val, lr_probs_val)
roc_lr = roc_auc_score(y_val, lr_probs_val)
results['Logistic Regression'] = {'val_prauc': ap_lr, 'val_rocauc': roc_lr,
                                    'probs_val': lr_probs_val, 'probs_test': lr_probs_test}
print(f"        Val PR-AUC: {ap_lr:.4f} | ROC-AUC: {roc_lr:.4f}")

# ── Model 2: Random Forest
print("    [2/4] Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             max_depth=12, min_samples_leaf=5,
                             n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
rf_probs_val = rf.predict_proba(X_val)[:, 1]
rf_probs_test = rf.predict_proba(X_test_proc)[:, 1]
ap_rf = average_precision_score(y_val, rf_probs_val)
roc_rf = roc_auc_score(y_val, rf_probs_val)
results['Random Forest'] = {'val_prauc': ap_rf, 'val_rocauc': roc_rf,
                             'probs_val': rf_probs_val, 'probs_test': rf_probs_test}
print(f"        Val PR-AUC: {ap_rf:.4f} | ROC-AUC: {roc_rf:.4f}")

# ── Model 3: XGBoost
print("    [3/4] XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='aucpr',
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              verbose=False)
xgb_probs_val = xgb_model.predict_proba(X_val)[:, 1]
xgb_probs_test = xgb_model.predict_proba(X_test_proc)[:, 1]
ap_xgb = average_precision_score(y_val, xgb_probs_val)
roc_xgb = roc_auc_score(y_val, xgb_probs_val)
results['XGBoost'] = {'val_prauc': ap_xgb, 'val_rocauc': roc_xgb,
                       'probs_val': xgb_probs_val, 'probs_test': xgb_probs_test}
print(f"        Val PR-AUC: {ap_xgb:.4f} | ROC-AUC: {roc_xgb:.4f}")

# ── Model 4: LightGBM (final model)
print("    [4/4] LightGBM (final model)...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=700, learning_rate=0.03, max_depth=7,
    num_leaves=63, scale_pos_weight=scale_pos_weight,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
lgb_probs_val = lgb_model.predict_proba(X_val)[:, 1]
lgb_probs_test = lgb_model.predict_proba(X_test_proc)[:, 1]
ap_lgb = average_precision_score(y_val, lgb_probs_val)
roc_lgb = roc_auc_score(y_val, lgb_probs_val)
results['LightGBM'] = {'val_prauc': ap_lgb, 'val_rocauc': roc_lgb,
                        'probs_val': lgb_probs_val, 'probs_test': lgb_probs_test}
print(f"        Val PR-AUC: {ap_lgb:.4f} | ROC-AUC: {roc_lgb:.4f}")

# Save final model
joblib.dump(lgb_model, MODEL_DIR / "model.pkl")
print("    LightGBM model saved.")

# ── Model comparison table
print("\n    === MODEL COMPARISON ===")
comp_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Val PR-AUC': [results[m]['val_prauc'] for m in results],
    'Val ROC-AUC': [results[m]['val_rocauc'] for m in results],
}).sort_values('Val PR-AUC', ascending=False)
print(comp_df.to_string(index=False))
comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# ─────────────────────────────────────────────
# SECTION 8: PR Curve & Threshold Selection
# ─────────────────────────────────────────────
print("\n[7] PR curve & threshold selection...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0f0f1a')

model_colors = {'Logistic Regression': '#00d4ff', 'Random Forest': '#7c3aed',
                'XGBoost': '#ffaa00', 'LightGBM': '#00ff88'}

ax = axes[0]
ax.set_facecolor('#1a1a2e')
for mname, mres in results.items():
    p, r, thresholds = precision_recall_curve(y_val, mres['probs_val'])
    ax.plot(r, p, color=model_colors[mname], linewidth=2,
            label=f"{mname} (PR-AUC={mres['val_prauc']:.3f})")
ax.set_xlabel('Recall', color='white', fontsize=12)
ax.set_ylabel('Precision', color='white', fontsize=12)
ax.set_title('Precision-Recall Curves (Validation)', color='white', fontsize=13, fontweight='bold')
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax.tick_params(colors='white')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.axhline(y_val.mean(), color='gray', linestyle='--', alpha=0.5, label='Baseline')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')

# ROC curves
ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')
for mname, mres in results.items():
    fpr, tpr, _ = roc_curve(y_val, mres['probs_val'])
    ax2.plot(fpr, tpr, color=model_colors[mname], linewidth=2,
             label=f"{mname} (ROC={mres['val_rocauc']:.3f})")
ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('False Positive Rate', color='white', fontsize=12)
ax2.set_ylabel('True Positive Rate', color='white', fontsize=12)
ax2.set_title('ROC Curves (Validation)', color='white', fontsize=13, fontweight='bold')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig(FIG_DIR / "05_pr_roc_curves.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 05_pr_roc_curves.png")

# Threshold selection: maximize F1 or Recall >= 0.90
p_lgb, r_lgb, thresh_lgb = precision_recall_curve(y_val, lgb_probs_val)
f1_scores = 2 * p_lgb * r_lgb / (p_lgb + r_lgb + 1e-9)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresh_lgb[best_idx]
best_f1 = f1_scores[best_idx]
best_prec = p_lgb[best_idx]
best_rec  = r_lgb[best_idx]
print(f"    Best threshold: {best_threshold:.4f}")
print(f"    At threshold  → Precision: {best_prec:.3f} | Recall: {best_rec:.3f} | F1: {best_f1:.3f}")

# ─────────────────────────────────────────────
# SECTION 9: Final Evaluation on Test Set
# ─────────────────────────────────────────────
print("\n[8] Evaluating on test set with LightGBM...")
lgb_test_preds = (lgb_probs_test >= best_threshold).astype(int)
test_prauc  = average_precision_score(y_test, lgb_probs_test)
test_rocauc = roc_auc_score(y_test, lgb_probs_test)
test_f1     = f1_score(y_test, lgb_test_preds)
cm = confusion_matrix(y_test, lgb_test_preds)

print(f"    Test PR-AUC : {test_prauc:.4f}")
print(f"    Test ROC-AUC: {test_rocauc:.4f}")
print(f"    Test F1     : {test_f1:.4f}")
print(f"    Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, lgb_test_preds, target_names=['Non-APS','APS Fail'])}")

# ── Confusion Matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#1a1a2e')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-APS (0)', 'APS Fail (1)'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix @ threshold={best_threshold:.2f}\n(LightGBM on Test Set)',
             color='white', fontsize=13, fontweight='bold', pad=15)
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
for text in disp.text_.ravel():
    text.set_color('white')
    text.set_fontsize(14)
    text.set_fontweight('bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "06_confusion_matrix.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 06_confusion_matrix.png")

# ─────────────────────────────────────────────
# SECTION 10: SHAP Feature Importance
# ─────────────────────────────────────────────
print("\n[9] SHAP feature importance...")
try:
    explainer = shap.TreeExplainer(lgb_model)
    # Use a sample for speed
    sample_idx = np.random.choice(len(X_train_proc), min(1000, len(X_train_proc)), replace=False)
    shap_values = explainer.shap_values(X_train_proc[sample_idx])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top20_shap_idx = np.argsort(shap_abs_mean)[-20:][::-1]
    top20_shap_names = [all_feature_cols[i] for i in top20_shap_idx]
    top20_shap_vals  = shap_abs_mean[top20_shap_idx]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    gradient_colors = plt.cm.plasma(np.linspace(0.3, 0.9, 20))
    bars = ax.barh(range(20), top20_shap_vals[::-1],
                   color=gradient_colors[::-1], edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20_shap_names[::-1], fontsize=10, color='white')
    ax.set_xlabel('Mean |SHAP Value|', color='white', fontsize=12)
    ax.set_title('Top 20 Features by SHAP Importance (LightGBM)', color='white',
                 fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_shap_importance.png", dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    print("    Saved: 07_shap_importance.png")
    shap_success = True
except Exception as e:
    print(f"    SHAP failed: {e} — using LightGBM built-in importance")
    shap_success = False

# LightGBM built-in feature importance (always)
fi = lgb_model.feature_importances_
top20_fi_idx = np.argsort(fi)[-20:][::-1]
top20_fi_names = [all_feature_cols[i] for i in top20_fi_idx]
top20_fi_vals  = fi[top20_fi_idx]

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#1a1a2e')
gradient_colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
ax.barh(range(20), top20_fi_vals[::-1], color=gradient_colors[::-1],
        edgecolor='white', linewidth=0.3)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_fi_names[::-1], fontsize=10, color='white')
ax.set_xlabel('Feature Importance (Split Count)', color='white', fontsize=12)
ax.set_title('Top 20 Features — LightGBM Importance', color='white',
             fontsize=14, fontweight='bold', pad=15)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')
plt.tight_layout()
plt.savefig(FIG_DIR / "08_feature_importance.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 08_feature_importance.png")

# ─────────────────────────────────────────────
# SECTION 11: Risk Segmentation
# ─────────────────────────────────────────────
print("\n[10] Risk segmentation...")

def assign_risk(prob):
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    else:
        return "Low"

risk_buckets = [assign_risk(p) for p in lgb_probs_test]
bucket_counts = pd.Series(risk_buckets).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0f0f1a')

ax = axes[0]
ax.set_facecolor('#1a1a2e')
bucket_order = ['High', 'Medium', 'Low']
colors_risk = {'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#00d4ff'}
bc = [bucket_counts.get(b, 0) for b in bucket_order]
bars = ax.bar(bucket_order, bc, color=[colors_risk[b] for b in bucket_order],
              edgecolor='white', linewidth=0.5, width=0.5)
ax.set_title('Risk Bucket Distribution (Test Set)', color='white', fontsize=13, fontweight='bold')
ax.set_ylabel('Truck Count', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')
for bar, count in zip(bars, bc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count:,}', ha='center', va='bottom', color='white', fontweight='bold')

# Risk bucket vs actual failures
ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')
risk_df = pd.DataFrame({'risk': risk_buckets, 'actual': y_test.values})
for i, b in enumerate(bucket_order):
    sub = risk_df[risk_df['risk'] == b]
    fail_rate = sub['actual'].mean() * 100 if len(sub) > 0 else 0
    ax2.bar(b, fail_rate, color=colors_risk[b], edgecolor='white', linewidth=0.5, width=0.5, alpha=0.9)
    ax2.text(i, fail_rate + 0.5, f'{fail_rate:.1f}%', ha='center', va='bottom',
             color='white', fontweight='bold', fontsize=12)
ax2.set_title('Actual APS Failure Rate by Risk Bucket', color='white', fontsize=13, fontweight='bold')
ax2.set_ylabel('APS Failure Rate (%)', color='white', fontsize=12)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig(FIG_DIR / "09_risk_segmentation.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 09_risk_segmentation.png")

# ─────────────────────────────────────────────
# SECTION 12: KPI Summary
# ─────────────────────────────────────────────
print("\n[11] KPI Summary...")
tn, fp, fn, tp = cm.ravel()
high_risk_df = risk_df[risk_df['risk'] == 'High']

kpi_data = {
    'Metric': [
        'Overall Failure Rate (Test)',
        'Test PR-AUC',
        'Test ROC-AUC',
        'Test F1 Score',
        'Test Recall (at threshold)',
        'Test Precision (at threshold)',
        'True Positives (caught failures)',
        'False Negatives (missed failures)',
        'False Positives (unnecessary inspections)',
        'High-Risk Trucks',
        'High-Risk Failure Rate',
        'Optimal Threshold',
    ],
    'Value': [
        f"{y_test.mean()*100:.2f}%",
        f"{test_prauc:.4f}",
        f"{test_rocauc:.4f}",
        f"{test_f1:.4f}",
        f"{tp/(tp+fn):.4f}",
        f"{tp/(tp+fp):.4f}",
        str(tp),
        str(fn),
        str(fp),
        str(bc[0]),
        f"{high_risk_df['actual'].mean()*100:.1f}%" if len(high_risk_df) > 0 else "N/A",
        f"{best_threshold:.4f}",
    ]
}
kpi_df = pd.DataFrame(kpi_data)
print(kpi_df.to_string(index=False))
kpi_df.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)

# ─────────────────────────────────────────────
# SECTION 13: Output File
# ─────────────────────────────────────────────
print("\n[12] Exporting predictions...")
out_df = pd.DataFrame({
    'truck_id': range(len(y_test)),
    'actual_class': y_test.values,
    'predicted_probability': lgb_probs_test.round(6),
    'predicted_class': lgb_test_preds,
    'risk_bucket': risk_buckets,
    'recommendation': [
        'Immediate inspection' if r == 'High' else
        'Schedule inspection within 48h' if r == 'Medium' else
        'Routine check'
        for r in risk_buckets
    ]
})
out_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
out_df.to_excel(OUTPUT_DIR / "test_predictions.xlsx", index=False)
print(f"    Saved {len(out_df)} predictions to outputs/test_predictions.csv + .xlsx")

# ─────────────────────────────────────────────
# SECTION 14: Error Analysis
# ─────────────────────────────────────────────
print("\n[13] Error analysis...")
fn_df = out_df[(out_df['actual_class'] == 1) & (out_df['predicted_class'] == 0)]
fp_df = out_df[(out_df['actual_class'] == 0) & (out_df['predicted_class'] == 1)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0f0f1a')

ax = axes[0]
ax.set_facecolor('#1a1a2e')
ax.hist(fn_df['predicted_probability'], bins=20, color='#ff6b6b', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.set_title(f'False Negatives (n={len(fn_df)})\nMissed APS Failures', color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Probability', color='white', fontsize=11)
ax.set_ylabel('Count', color='white', fontsize=11)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')
ax.axvline(best_threshold, color='yellow', linestyle='--', label=f'Threshold={best_threshold:.2f}')
ax.legend(facecolor='#1a1a2e', labelcolor='white')

ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')
ax2.hist(fp_df['predicted_probability'], bins=20, color='#ffaa00', edgecolor='white', linewidth=0.5, alpha=0.85)
ax2.set_title(f'False Positives (n={len(fp_df)})\nUnnecessary Inspections', color='white', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted Probability', color='white', fontsize=11)
ax2.set_ylabel('Count', color='white', fontsize=11)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#333355')
ax2.axvline(best_threshold, color='yellow', linestyle='--', label=f'Threshold={best_threshold:.2f}')
ax2.legend(facecolor='#1a1a2e', labelcolor='white')

plt.tight_layout()
plt.savefig(FIG_DIR / "10_error_analysis.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 10_error_analysis.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"  Final model : LightGBM")
print(f"  Test PR-AUC : {test_prauc:.4f}")
print(f"  Test ROC-AUC: {test_rocauc:.4f}")
print(f"  Test F1     : {test_f1:.4f}")
print(f"  Threshold   : {best_threshold:.4f}")
print(f"  TP={tp} | FP={fp} | TN={tn} | FN={fn}")
print(f"  Missed failures (FN): {fn} out of {tp+fn} actual APS failures")
print("="*60)
print("\nAll outputs saved to:")
print(f"  Figures       : {FIG_DIR}")
print(f"  Predictions   : {OUTPUT_DIR / 'test_predictions.csv'}")
print(f"  KPI Summary   : {OUTPUT_DIR / 'kpi_summary.csv'}")
print(f"  Model         : {MODEL_DIR / 'model.pkl'}")
print(f"  Preprocessor  : {MODEL_DIR / 'preprocessor.pkl'}")
print("\nDone!")

# Save metadata for API
import json
meta = {
    'best_threshold': float(best_threshold),
    'test_prauc': float(test_prauc),
    'test_rocauc': float(test_rocauc),
    'test_f1': float(test_f1),
    'feature_cols': keep_cols,
    'high_miss_cols': high_miss_cols,
    'all_feature_cols': all_feature_cols,
    'label_map': {'pos': 1, 'neg': 0},
    'top_features': top20_fi_names[:10]
}
with open(MODEL_DIR / 'model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("  Metadata      : api/model_artifacts/model_meta.json")
