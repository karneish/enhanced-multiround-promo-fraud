"""Entry point: serves the dashboard (frontend) plus the Flask API.

Run with::

    python run.py

then open http://127.0.0.1:5050
"""

import os

from flask import send_from_directory

from backend.app import app

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')


@app.get('/')
def index():
    return send_from_directory(FRONTEND, 'index.html')


@app.get('/<path:path>')
def assets(path):
    return send_from_directory(FRONTEND, path)


if __name__ == '__main__':
    print('Intelligent Fraud Generator dashboard -> http://127.0.0.1:5050')
    app.run(host='127.0.0.1', port=5050, threaded=True, debug=False)
