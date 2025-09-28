# Importar librerías
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar app
app = Flask(__name__)

# Carga del modelo al iniciar la app
MODEL_FILENAME = 'modelo_cancer.pkl'
model = None
if os.path.exists(MODEL_FILENAME):
    try:
        model = joblib.load(MODEL_FILENAME)
        logger.info("Modelo cargado: %s", MODEL_FILENAME)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("No se pudo cargar el modelo: %s", exc)
else:
    logger.warning("Archivo de modelo no encontrado: %s", MODEL_FILENAME)

# Endpoint de ruta principal
@app.route('/', methods=['GET'])
def home():
    """Devuelve el estado de la API y del modelo."""
    status = 'ok' if model is not None else 'model-missing'
    code = 200 if model is not None else 500
    return jsonify({'status': status}), code

# Endpoint para realizar predicción
@app.route('/predict', methods=['POST'])
def predict():
    """Realiza una predicción a partir de un JSON con 'features'."""
    try:
        # Verificar modelo cargado
        if model is None:
            error_msg = 'Modelo no disponible'
            logger.error('error: %s', error_msg)
            return jsonify({'error': error_msg}), 500

        # Verificar tipo de contenido
        if not request.is_json:
            error_msg = 'Content-Type debe ser application/json'
            logger.error('error: %s', error_msg)
            return jsonify({ 'error': error_msg}), 400

        data = request.get_json(force=True)
        # Chequear que exista la clave "features"
        if 'features' not in data:
            error_msg = 'Falta la clave "features" en los datos de entrada'
            logger.error('error: %s', error_msg)
            return jsonify({'error': error_msg}), 400

        features = data['features']

        # Chequear que haya exactamente 4 números reales en la variable features
        if len(features) != 30 or not all(isinstance(x, (int, float)) for x in features):
            error_msg = 'Favor revisar que hayan 30 números reales en features.'
            logger.error('error: %s', error_msg)
            return jsonify({'error': error_msg}), 400

        # Convertir a array
        input_array = np.array(features).reshape(1, -1)
        # Realizar predicción
        prediction = model.predict(input_array)
        # Retornar la predicción de la clase
        logger.info('Predicción realizada: %d', int(prediction[0]))
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        logger.error('error: %s', str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)