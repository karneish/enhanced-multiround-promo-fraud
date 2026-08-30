from flask import Flask
from flask_cors import CORS

from backend.config import Config
from backend.routes.experiments import experiments_bp
from backend.routes.results import results_bp
from backend.routes.models import models_bp


def create_app(config=None):
    app = Flask(__name__)

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    try:
        app.json.sort_keys = False
    except AttributeError:
        app.config['JSON_SORT_KEYS'] = False

    if config:
        app.config.update(config)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(experiments_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(models_bp)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        import torch
        return {
            'status': 'ok',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_available': torch.cuda.is_available(),
        }

    @app.route('/api', methods=['GET'])
    def api_info():
        return {
            'name': 'Adaptive Multi-Model Fraud Detection API',
            'version': '1.0.0',
            'endpoints': {
                'health': 'GET /api/health',
                'models': 'GET /api/models',
                'datasets': 'GET /api/datasets',
                'configs': 'GET /api/configs',
                'experiments': 'GET|POST /api/experiments',
                'experiment_detail': 'GET /api/experiments/<id>',
                'experiment_status': 'GET /api/experiments/<id>/status',
                'results': 'GET /api/results',
                'result_file': 'GET /api/results/<path>/<filename>',
            }
        }

    @app.errorhandler(404)
    def not_found(e):
        return {'error': 'Not found'}, 404

    @app.errorhandler(400)
    def bad_request(e):
        return {'error': 'Bad request'}, 400

    @app.errorhandler(500)
    def internal_error(e):
        return {'error': 'Internal server error'}, 500

    return app
