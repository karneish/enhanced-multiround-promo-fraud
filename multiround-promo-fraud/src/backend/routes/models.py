from flask import Blueprint, jsonify

from backend.services.experiment_service import ExperimentService

models_bp = Blueprint('models', __name__)
service = ExperimentService()


@models_bp.route('/api/models', methods=['GET'])
def list_models():
    models = service.list_models()
    return jsonify({'models': models, 'count': len(models)})


@models_bp.route('/api/datasets', methods=['GET'])
def list_datasets():
    datasets = service.list_datasets()
    return jsonify({'datasets': datasets, 'count': len(datasets)})


@models_bp.route('/api/configs', methods=['GET'])
def list_configs():
    configs = service.list_available_configs()
    return jsonify({'configs': configs, 'count': len(configs)})
