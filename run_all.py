"""Run all three dashboards behind a single URL (http://127.0.0.1:5050)."""

import atexit
import os
import subprocess
import sys

from flask import Flask

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, 'multiround-promo-fraud', '.venv', 'Scripts', 'python.exe')
PYTHON = VENV_PY if os.path.isfile(VENV_PY) else sys.executable

APPS = [
    ('Main Dashboard (research framework)', 'multiround-promo-fraud\\dashboard', 'app.py', 5051),
    ('Adaptive Defensive Layer (ADL)', 'adaptive-defensive-layer', 'run.py', 5052),
    ('Intelligent Fraud Generator (demo)', 'intelligent-fraud-generator', 'run.py', 5053),
]

PROCS = []


def start(desc, rel_cwd, script, port):
    cwd = os.path.join(ROOT, rel_cwd)
    env = os.environ.copy()
    env['PORT'] = str(port)
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
    p = subprocess.Popen([PYTHON, script], cwd=cwd, env=env, **kwargs)
    PROCS.append(p)
    print(f'  [{desc}] http://127.0.0.1:{port}  (pid {p.pid})')


def shutdown():
    for p in PROCS:
        try:
            p.terminate()
        except Exception:
            pass


app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Round Promo Fraud - All Dashboards</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Segoe UI, Arial, sans-serif; background: #0f1420; color: #e8eef7; }
  header { display: flex; align-items: center; justify-content: space-between;
           padding: 14px 24px; background: #1a2233; border-bottom: 1px solid #2a3650; }
  header h1 { font-size: 17px; font-weight: 600; }
  header .links a { color: #7fb3ff; text-decoration: none; margin-left: 16px; font-size: 13px; }
  header .links a:hover { text-decoration: underline; }
  main { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; min-height: calc(100vh - 54px); }
  section { display: flex; flex-direction: column; border-right: 1px solid #2a3650; min-height: 0; }
  section:last-child { border-right: none; }
  .label { padding: 10px 14px; font-size: 12px; font-weight: 600; background: #161e2e;
           color: #9fb3d1; border-bottom: 1px solid #2a3650; text-transform: uppercase; letter-spacing: .5px; }
  iframe { flex: 1; width: 100%; border: 0; min-height: 0; background: #fff; }
</style>
</head>
<body>
<header>
  <h1>Multi-Round Promo Fraud Detection - All Dashboards</h1>
  <div class="links">
    <a href="http://127.0.0.1:5051" target="_blank">Main framework</a>
    <a href="http://127.0.0.1:5052" target="_blank">ADL</a>
    <a href="http://127.0.0.1:5053" target="_blank">Intelligent Generator</a>
  </div>
</header>
<main>
  <section><div class="label">Main Framework</div><iframe src="http://127.0.0.1:5051"></iframe></section>
  <section><div class="label">Adaptive Defensive Layer</div><iframe src="http://127.0.0.1:5052"></iframe></section>
  <section><div class="label">Intelligent Fraud Generator</div><iframe src="http://127.0.0.1:5053"></iframe></section>
</main>
</body>
</html>"""


@app.route('/')
def index():
    return PAGE


if __name__ == '__main__':
    print('Starting all dashboards...')
    for a in APPS:
        start(*a)
    atexit.register(shutdown)
    print('All dashboards in one place ->  http://127.0.0.1:5050')
    app.run(host='127.0.0.1', port=5050, threaded=True)
