import json, os
from pathlib import Path
import pandas as pd

BASE = Path(r'c:\Users\prane\Downloads\UP-Analytics-2026-Assessment-PraneethRamisetti')

checks = []

def check(label, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    checks.append((status, label, detail))

# --- Files
check('notebook.ipynb exists',         (BASE/'notebook.ipynb').exists())
check('run_analysis.py exists',        (BASE/'run_analysis.py').exists())
check('requirements.txt exists',       (BASE/'requirements.txt').exists())
check('README.md exists',              (BASE/'README.md').exists())
check('api/main.py exists',            (BASE/'api/main.py').exists())
check('api/schemas.py exists',         (BASE/'api/schemas.py').exists())
check('model.pkl exists',              (BASE/'api/model_artifacts/model.pkl').exists())
check('preprocessor.pkl exists',       (BASE/'api/model_artifacts/preprocessor.pkl').exists())
check('model_meta.json exists',        (BASE/'api/model_artifacts/model_meta.json').exists())
check('feature_config.pkl exists',     (BASE/'api/model_artifacts/feature_config.pkl').exists())
check('test_predictions.csv exists',   (BASE/'outputs/test_predictions.csv').exists())
check('test_predictions.xlsx exists',  (BASE/'outputs/test_predictions.xlsx').exists())
check('kpi_summary.csv exists',        (BASE/'outputs/kpi_summary.csv').exists())
check('model_comparison.csv exists',   (BASE/'outputs/model_comparison.csv').exists())

# --- Figures
figs = list((BASE/'outputs/figures').glob('*.png'))
check('>=8 EDA figures exist', len(figs) >= 8, f'{len(figs)} figures found')

# --- Screenshots
shots = list((BASE/'screenshots').glob('*.png'))
check('API screenshots exist', len(shots) >= 2, f'{len(shots)} screenshots found')

# --- Git
check('Git repo initialized', (BASE/'.git').exists())
check('.gitignore exists',    (BASE/'.gitignore').exists())

# --- Predictions content
preds = pd.read_csv(BASE/'outputs/test_predictions.csv')
check('test_predictions has 16000 rows',   len(preds)==16000,           f'{len(preds)} rows')
check('predictions have risk_bucket col',   'risk_bucket' in preds.columns)
check('predictions have probability col',   'predicted_probability' in preds.columns)
check('predictions have recommendation col','recommendation' in preds.columns)

# --- KPI
kpi = pd.read_csv(BASE/'outputs/kpi_summary.csv')
check('kpi_summary has >= 5 metrics', len(kpi) >= 5, f'{len(kpi)} metrics')

# --- Model comparison
comp = pd.read_csv(BASE/'outputs/model_comparison.csv')
check('model_comparison has >= 3 models', len(comp) >= 3, f'{len(comp)} models compared')

# --- Notebook cells
with open(BASE/'notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cell_count = len(nb['cells'])
check('notebook has >= 40 cells', cell_count >= 40, f'{cell_count} cells')
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells   = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
check('notebook has >= 15 code cells',     len(code_cells) >= 15, f'{len(code_cells)} code cells')
check('notebook has >= 10 markdown cells', len(md_cells)   >= 10, f'{len(md_cells)} markdown cells')

# --- Model meta
with open(BASE/'api/model_artifacts/model_meta.json') as f:
    meta = json.load(f)
check('model_meta has threshold',       'best_threshold' in meta,           f"threshold={meta.get('best_threshold', 'MISSING')}")
check('model PR-AUC > 0.80',            meta.get('test_prauc', 0) > 0.80,   f"PR-AUC={meta.get('test_prauc', 0):.4f}")
check('model ROC-AUC > 0.95',          meta.get('test_rocauc', 0) > 0.95,  f"ROC-AUC={meta.get('test_rocauc', 0):.4f}")
check('model has top_features list',    len(meta.get('top_features',[])) > 0)

# --- README length check
readme_txt   = (BASE/'README.md').read_text(encoding='utf-8')
readme_lines = len(readme_txt.splitlines())
check('README <= 160 lines (~2 printed pages)', readme_lines <= 160, f'{readme_lines} lines')
check('README has AI usage disclosure', 'AI Usage' in readme_txt or 'AI usage' in readme_txt)
check('README has setup instructions',  'pip install' in readme_txt or 'requirements' in readme_txt.lower())

# --- API files
api_txt = (BASE/'api/main.py').read_text(encoding='utf-8')
check('API has /predict endpoint',       '/predict' in api_txt)
check('API has /predict/batch endpoint', '/predict/batch' in api_txt)
check('API has health endpoint',         '/health' in api_txt or 'GET /' in api_txt)
check('API has CORS middleware',         'CORSMiddleware' in api_txt)

# --- requirements.txt has key packages
req_txt = (BASE/'requirements.txt').read_text()
for pkg in ['lightgbm', 'xgboost', 'scikit-learn', 'fastapi', 'shap', 'pandas']:
    check(f'requirements.txt has {pkg}', pkg.lower() in req_txt.lower())

# --- FINAL REPORT
print()
print('=' * 65)
print('  FULL DELIVERABLES VERIFICATION REPORT')
print('  APS Failure Prediction - Scania Trucks')
print('=' * 65)
passes = fails = 0
for status, label, detail in checks:
    icon = 'OK  ' if status == 'PASS' else 'FAIL'
    note = f'  [{detail}]' if detail else ''
    print(f'  [{icon}]  {label}{note}')
    if status == 'PASS': passes += 1
    else: fails += 1
print('=' * 65)
print(f'  RESULT: {passes}/{passes+fails} checks PASSED  |  {fails} FAILED')
print('=' * 65)
if fails == 0:
    print('  ALL CHECKS PASSED - Submission Ready!')
else:
    print('  Some items need attention (see FAIL above)')
print()
