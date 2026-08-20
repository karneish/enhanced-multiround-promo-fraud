"""Entry point: serves frontend + Flask API."""
import os
import sys
from backend.app import app

PORT = int(os.environ.get('PORT', 5050))

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    print(f'Adaptive Ensemble Detector dashboard -> http://127.0.0.1:{PORT}')
    app.run(host='127.0.0.1', port=PORT, threaded=True)
