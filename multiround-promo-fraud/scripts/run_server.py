import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.app import create_app
from backend.config import Config

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Adaptive Fraud Detection API Server')
    parser.add_argument('--host', type=str, default=Config.HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=Config.PORT, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    app = create_app()

    print(f'Starting Adaptive Fraud Detection API on {args.host}:{args.port}')
    print(f'Device: cpu')
    print(f'Result directory: {Config.RESULT_DIR}')
    print(f'Dataset directory: {Config.DATASET_DIR}')

    app.run(host=args.host, port=args.port, debug=args.debug)
