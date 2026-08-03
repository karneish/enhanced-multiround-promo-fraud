"""Entry point: serves the ADL dashboard (frontend) plus the Flask API.

Run with::

    python run.py

then open http://127.0.0.1:5050

The port can be overridden with the ``PORT`` environment variable.
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
    port = int(os.environ.get('PORT', '5050'))
    print(f'Adaptive Defensive Layer dashboard -> http://127.0.0.1:{port}')
    app.run(host='127.0.0.1', port=port, threaded=True, debug=False)
