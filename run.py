"""Unified launcher: starts all 4 Flask backends + Vite dev server.

Usage:
    python run.py          # starts everything, opens browser
    python run.py --no-browser  # skip auto-open
"""
import atexit
import os
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, 'multiround-promo-fraud', '.venv', 'Scripts', 'python.exe')
PYTHON = VENV_PY if os.path.isfile(VENV_PY) else sys.executable
FRONTEND_DIR = os.path.join(ROOT, 'frontend')

APPS = [
    ('Main Dashboard',  'multiround-promo-fraud\\dashboard', 'app.py', 5051),
    ('Adaptive Defensive Layer', 'adaptive-defensive-layer',   'run.py', 5052),
    ('Intelligent Fraud Generator', 'intelligent-fraud-generator', 'run.py', 5053),
    ('Adaptive Ensemble Detector', 'adaptive-ensemble-detector', 'run.py', 5054),
]

PROCS = []


def start_backend(desc, rel_cwd, script, port):
    cwd = os.path.join(ROOT, rel_cwd)
    env = os.environ.copy()
    env['PORT'] = str(port)
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
    try:
        p = subprocess.Popen([PYTHON, script], cwd=cwd, env=env, **kwargs)
        PROCS.append(p)
        print(f'  [{desc}] http://127.0.0.1:{port}  (pid {p.pid})')
    except FileNotFoundError:
        print(f'  [{desc}] SKIP - python not found at {PYTHON}')


def start_frontend():
    npm = 'npm.cmd' if sys.platform == 'win32' else 'npm'
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
    try:
        p = subprocess.Popen([npm, 'run', 'dev'], cwd=FRONTEND_DIR, **kwargs)
        PROCS.append(p)
        print(f'  [React Frontend] http://127.0.0.1:3000  (pid {p.pid})')
    except FileNotFoundError:
        print('  [React Frontend] SKIP - npm not found. Run: cd frontend && npm install')


def shutdown():
    for p in PROCS:
        try:
            p.terminate()
        except Exception:
            pass


if __name__ == '__main__':
    no_browser = '--no-browser' in sys.argv
    print('=' * 60)
    print('  Multi-Round Promo Fraud Detection - Unified Dashboard')
    print('=' * 60)
    print('\nStarting backends...')
    for desc, cwd, script, port in APPS:
        start_backend(desc, cwd, script, port)
    print('\nStarting frontend...')
    start_frontend()
    atexit.register(shutdown)
    print('\n' + '=' * 60)
    print('  Dashboard: http://127.0.0.1:3000')
    print('=' * 60)
    if not no_browser:
        time.sleep(3)
        try:
            webbrowser.open('http://127.0.0.1:3000')
        except Exception:
            pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nShutting down...')
