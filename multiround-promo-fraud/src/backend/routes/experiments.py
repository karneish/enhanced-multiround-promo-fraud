from flask import Blueprint, request, jsonify, send_file

from backend.services.experiment_service import ExperimentService

experiments_bp = Blueprint('experiments', __name__)
service = ExperimentService()


@experiments_bp.route('/api/experiments', methods=['GET'])
def list_experiments():
    experiments = service.list_experiments()
    return jsonify({'experiments': experiments, 'count': len(experiments)})


@experiments_bp.route('/api/experiments', methods=['POST'])
def create_experiment():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    required_fields = ['dataset', 'model_name', 'round_num']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    experiment_id = service.start_experiment(data)
    return jsonify({'experiment_id': experiment_id, 'status': 'queued'}), 201


@experiments_bp.route('/api/experiments/<experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    result = service.get_experiment(experiment_id)
    if result is None:
        return jsonify({'error': 'Experiment not found'}), 404
    return jsonify(result)


@experiments_bp.route('/api/experiments/<experiment_id>/status', methods=['GET'])
def get_experiment_status(experiment_id):
    result = service.get_experiment_status(experiment_id)
    if result is None:
        return jsonify({'error': 'Experiment not found'}), 404
    return jsonify(result)
