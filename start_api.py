"""
APS Failure Prediction API — Startup script
Run from project root: python start_api.py
"""
import subprocess
import sys

print("Starting APS Failure Prediction API...")
print("Swagger UI: http://localhost:8000/docs")
print("ReDoc:      http://localhost:8000/redoc")
print("Press Ctrl+C to stop.\n")

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "api.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload"
])
