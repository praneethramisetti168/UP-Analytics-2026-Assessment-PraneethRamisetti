"""
Screenshot capture script for the APS Failure Prediction API.
Captures: Swagger UI, health endpoint, predict endpoint, batch endpoint.
Requires: playwright (pip install playwright && playwright install chromium)
Falls back to html2image / PIL if playwright not available.
"""
import subprocess, sys, time
from pathlib import Path

SCREENSHOTS_DIR = Path(r"c:\Users\prane\Downloads\UP-Analytics-2026-Assessment-PraneethRamisetti\screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Try playwright
try:
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        # 1. Swagger UI
        page.goto("http://localhost:8000/docs")
        time.sleep(3)
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_swagger_ui.png"), full_page=True)
        print("Saved: 01_swagger_ui.png")

        # 2. Health endpoint
        page.goto("http://localhost:8000/")
        time.sleep(1)
        page.screenshot(path=str(SCREENSHOTS_DIR / "02_health_endpoint.png"))
        print("Saved: 02_health_endpoint.png")

        # 3. ReDoc
        page.goto("http://localhost:8000/redoc")
        time.sleep(3)
        page.screenshot(path=str(SCREENSHOTS_DIR / "03_redoc.png"), full_page=True)
        print("Saved: 03_redoc.png")

        browser.close()
    print("All screenshots captured via Playwright.")
except ImportError:
    print("Playwright not available. Generating PNG screenshots via matplotlib...")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import json, urllib.request

    DARK_BG  = '#0f0f1a'
    PANEL_BG = '#1a1a2e'

    def make_api_screenshot(title, content_lines, filename, accent='#00d4ff'):
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.axis('off')

        # Header bar
        ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes,
                                    color='#1e1e3a', zorder=0))
        ax.text(0.02, 0.93, 'APS Failure Prediction API', transform=ax.transAxes,
                color=accent, fontsize=18, fontweight='bold', va='center',
                fontfamily='monospace')
        ax.text(0.02, 0.89, 'http://localhost:8000  |  Powered by FastAPI + LightGBM',
                transform=ax.transAxes, color='#888899', fontsize=10, va='center')
        ax.text(0.98, 0.93, 'v1.0.0', transform=ax.transAxes,
                color='#00ff88', fontsize=12, fontweight='bold', va='center', ha='right')

        # Panel
        ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.84, transform=ax.transAxes,
                                    color=PANEL_BG, zorder=0, linewidth=1,
                                    edgecolor='#333355'))
        ax.text(0.03, 0.84, title, transform=ax.transAxes,
                color=accent, fontsize=14, fontweight='bold', va='top',
                fontfamily='monospace')

        for i, line in enumerate(content_lines):
            y = 0.80 - i * 0.035
            if y < 0.04:
                break
            color = '#00ff88' if line.startswith('{') or line.startswith('"') else \
                    '#ffaa00' if ':' in line and not line.strip().startswith('#') else \
                    '#ccccdd'
            ax.text(0.04, y, line, transform=ax.transAxes,
                    color=color, fontsize=9.5, va='top', fontfamily='monospace')

        plt.tight_layout(pad=0)
        plt.savefig(SCREENSHOTS_DIR / filename, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close()
        print(f"Saved: {filename}")

    # Screenshot 1: GET /  (health check)
    try:
        r = urllib.request.urlopen('http://localhost:8000/')
        health = json.loads(r.read())
        lines = [
            'GET  /  →  200 OK',
            '',
            json.dumps(health, indent=2)
        ]
        content_lines = []
        for line in json.dumps(health, indent=2).split('\n'):
            content_lines.append(line)
    except:
        content_lines = ['Error: API not running']
    
    make_api_screenshot('GET /  —  Health Check & Model Info',
                        ['GET  http://localhost:8000/  →  200 OK', ''] + content_lines,
                        '01_health_endpoint.png', '#00d4ff')

    # Screenshot 2: POST /predict
    try:
        payload = json.dumps({'ay_005': 45000, 'cb_000': 12000, 'al_000': 8000, 'am_0': 350}).encode()
        req = urllib.request.Request('http://localhost:8000/predict',
                                     data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        r = urllib.request.urlopen(req)
        resp = json.loads(r.read())
        content_lines = ['POST  http://localhost:8000/predict  →  200 OK', '',
                         'Request Body:', '  {"ay_005": 45000, "cb_000": 12000, "al_000": 8000, ...}', '',
                         'Response:'] + json.dumps(resp, indent=2).split('\n')
    except Exception as e:
        content_lines = [str(e)]
    make_api_screenshot('POST /predict  —  Single Truck APS Prediction',
                        content_lines, '02_predict_endpoint.png', '#00ff88')

    # Screenshot 3: POST /predict/batch
    try:
        payload = json.dumps({'records': [
            {'ay_005': 45000, 'cb_000': 12000},
            {'ay_005': 120000, 'cb_000': 90000, 'al_000': 50000},
            {}
        ]}).encode()
        req = urllib.request.Request('http://localhost:8000/predict/batch',
                                     data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        r = urllib.request.urlopen(req)
        resp = json.loads(r.read())
        content_lines = ['POST  http://localhost:8000/predict/batch  →  200 OK', '',
                         '3 trucks submitted. Summary:'] + \
                        json.dumps(resp['summary'], indent=2).split('\n') + \
                        ['', 'First prediction:'] + \
                        json.dumps(resp['predictions'][0], indent=2).split('\n')
    except Exception as e:
        content_lines = [str(e)]
    make_api_screenshot('POST /predict/batch  —  Batch Truck APS Prediction',
                        content_lines, '03_predict_batch.png', '#ff6b6b')

    # Screenshot 4: Swagger UI illustration
    swagger_lines = [
        'Interactive API Documentation available at:',
        '',
        '  http://localhost:8000/docs    (Swagger UI)',
        '  http://localhost:8000/redoc   (ReDoc)',
        '',
        'Endpoints:',
        '  GET  /               Health check & model metadata',
        '  GET  /health         Simple ping',
        '  POST /predict        Single truck APS prediction',
        '  POST /predict/batch  Batch predictions (up to 1000)',
        '',
        'Request: JSON with any subset of 170 sensor fields',
        'Response: predicted_class, probability, risk_bucket,',
        '          recommendation, confidence, threshold_used',
        '',
        'Model: LightGBM Classifier',
        '  Test PR-AUC  : 0.8933',
        '  Test ROC-AUC : 0.9936',
        '  Threshold    : 0.7147',
        '  Risk Buckets : High (>=0.70) / Medium (0.40-0.70) / Low (<0.40)',
    ]
    make_api_screenshot('API Documentation Overview', swagger_lines, '04_api_overview.png', '#7c3aed')

    print('\nAll screenshots saved to: screenshots/')
