from flask import Flask, request, jsonify
import json
import os
import numpy as np

app = Flask(__name__)

# Директория для сохранения данных
DATA_DIR = "data/facial"
os.makedirs(DATA_DIR, exist_ok=True)

@app.route('/save_facial_recording', methods=['POST'])
def save_facial_recording():
    """Сохраняет записанные данные blendshapes"""
    data = request.json
    patient_id = data.get('patientId')
    expression = data.get('expression')
    recorded_data = data.get('data', [])
    
    if not patient_id or not expression:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})
    
    # Создаем директорию для пациента
    patient_dir = os.path.join(DATA_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Сохраняем данные в JSON
    filepath = os.path.join(patient_dir, f"{expression}.json")
    with open(filepath, 'w') as f:
        json.dump(recorded_data, f, indent=2)
    
    return jsonify({
        'status': 'success',
        'saved_frames': len(recorded_data),
        'filepath': filepath
    })

@app.route('/upload_facial', methods=['POST'])
def upload_facial():
    """Загружает видеофайл"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'})
    
    file = request.files['file']
    upload_dir = os.path.join(DATA_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)
    
    return jsonify({'status': 'success', 'uploaded': file.filename})

@app.route('/facial_data_processing', methods=['POST'])
def facial_data_processing():
    """Обрабатывает данные: вычисляет статистику и производные"""
    data = request.json
    patient_id = data.get('patientId')
    expression = data.get('expression')
    
    if not patient_id or not expression:
        return jsonify({'status': 'error', 'message': 'Missing parameters'})
    
    # Загружаем сохраненные данные
    filepath = os.path.join(DATA_DIR, patient_id, f"{expression}.json")
    
    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': 'Recording not found'})
    
    with open(filepath, 'r') as f:
        recorded_data = json.load(f)
    
    if len(recorded_data) == 0:
        return jsonify({'status': 'error', 'message': 'No data in recording'})
    
    # Обрабатываем данные
    statistics = process_facial_data(recorded_data)
    
    # Сохраняем статистику
    stats_filepath = os.path.join(DATA_DIR, patient_id, f"{expression}_statistics.json")
    with open(stats_filepath, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    return jsonify({
        'status': 'success',
        'statistics': statistics,
        'processed_frames': len(recorded_data)
    })

def process_facial_data(recorded_data):
    """
    Обрабатывает записанные данные:
    1. Извлекает blendshapes (AU) для каждого фрейма
    2. Вычисляет производные
    3. Считает статистику: median, std, min, max
    """
    
    # Извлекаем все blendshapes по фреймам
    blendshapes_by_frame = []
    for frame in recorded_data:
        if 'blendshapes' in frame:
            blendshapes_by_frame.append(frame['blendshapes'])
    
    if len(blendshapes_by_frame) == 0:
        return {'error': 'No blendshapes data'}
    
    # Получаем все названия blendshapes
    blendshape_names = list(blendshapes_by_frame[0].keys())
    
    # 1. Статистика для blendshapes
    blendshapes_stats = {}
    for name in blendshape_names:
        values = [frame[name] for frame in blendshapes_by_frame if name in frame]
        if len(values) > 0:
            blendshapes_stats[name] = calculate_statistics(values)
    
    # 2. Вычисляем производные
    derivatives_by_frame = []
    for i in range(1, len(recorded_data)):
        dt = recorded_data[i]['timestamp'] - recorded_data[i-1]['timestamp']
        if dt == 0:
            continue
        
        derivatives = {}
        for name in blendshape_names:
            if name in recorded_data[i]['blendshapes'] and name in recorded_data[i-1]['blendshapes']:
                dvalue = recorded_data[i]['blendshapes'][name] - recorded_data[i-1]['blendshapes'][name]
                derivatives[name] = dvalue / dt
        
        derivatives_by_frame.append(derivatives)
    
    # 3. Статистика для производных
    derivatives_stats = {}
    if len(derivatives_by_frame) > 0:
        for name in blendshape_names:
            values = [frame[name] for frame in derivatives_by_frame if name in frame]
            if len(values) > 0:
                derivatives_stats[name] = calculate_statistics(values)
    
    return {
        'blendshapes': blendshapes_stats,
        'derivatives': derivatives_stats,
        'num_frames': len(recorded_data)
    }

def calculate_statistics(values):
    """Вычисляет median, std, min, max"""
    arr = np.array(values)
    
    return {
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
