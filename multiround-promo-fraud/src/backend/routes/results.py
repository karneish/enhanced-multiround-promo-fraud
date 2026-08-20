from flask import Blueprint, request, jsonify, send_file
import os

from backend.services.experiment_service import ExperimentService
from backend.config import Config

results_bp = Blueprint('results', __name__)
service = ExperimentService()


@results_bp.route('/api/results', methods=['GET'])
def list_results():
    results = service.list_results()
    return jsonify({'results': results, 'count': len(results)})


@results_bp.route('/api/results/<path:relative_path>/<filename>', methods=['GET'])
def get_result_file(relative_path, filename):
    if filename.endswith('.csv'):
        data = service.get_result_csv(relative_path, filename)
        if data is None:
            return jsonify({'error': 'File not found'}), 404
        return jsonify({'data': data, 'count': len(data)})

    elif filename.endswith(('.png', '.pdf', '.svg')):
        filepath = service.get_result_plot(relative_path, filename)
        if filepath is None:
            return jsonify({'error': 'File not found'}), 404
        return send_file(filepath)

    else:
        return jsonify({'error': 'Unsupported file type'}), 400
