"""
app.py

Author: Lander Combarro Exposito
Created: 2024/04/12
Last Modified: 2024/04/12

Iris Model Prediction API
-------------------------
This Flask API provides functionality for predicting the species of an Iris flower
based on its sepal and petal measurements. The model is a Logistic Regression 
trained on the Iris dataset.

Endpoints
---------
- GET /api/v1/predict: Predict the Iris flower species based on input features (sepal_length, sepal_width, petal_length, petal_width).
- GET /api/v1/retrain: Re-train the model with a new dataset and update the saved model.
- GET /api/v1/accuracy: Calculate and return the accuracy of the current trained model on the validation dataset. 
- POST /webhook: Update the model by pulling the latest changes from the GitHub repository (used for deployment).

The API also includes basic error handling for missing or invalid parameters, and ensures that the model is always available for predictions.
"""

# Libraries
import json
import os
import subprocess

from flask import Flask, jsonify, request
from sklearn.metrics import accuracy_score

from utils.iris_model import (download_data, load_or_initialize_model, load_data,
                              load_model, save_model, train_model)
from utils.variables import CLASS_MAPPING, DATA_PATH, MODEL_PATH

# Change the current working directory to the directory of this script
os.chdir(os.path.dirname(__file__))

# Create a Flask application instance and enable debug mode for development
app = Flask(__name__)
app.config['DEBUG'] = True

# Download data; and load or initialize model
download_data(data_path=DATA_PATH)
model = load_or_initialize_model(data_path=DATA_PATH, model_path=MODEL_PATH)

# Landing page route
@app.route('/', methods=['GET'])
def home():
    # Define the response dictionary with message and available endpoints
    response = {
        'message': 'Welcome to the Iris flower model prediction API',
        'endpoints': {
            '/api/v1/predict': 'Provides predictions based on input features (GET)',
            '/api/v1/retrain': 'Retrains the model with a new dataset (GET)',
            '/api/v1/accuracy': 'Shows the current accuracy of the model (GET)'
        },
        'example': {
            '/api/v1/predict?sepal_length=5.0&sepal_width=3.6&petal_length=1.4&petal_width=0.2': 
                'Add an endpoint like this one to predict the species of the Iris flower using the provided feature values'
        }
    }

    # Return the response as a JSON with proper formatting
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False, indent=4),
        status=200,
        mimetype='application/json'
    )

# Perform prediction
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except (TypeError, ValueError):
        return jsonify({'error': 
            'Please provide valid numeric values for all parameters: sepal_length, sepal_width, petal_length, and petal_width'
            }), 400

    try:
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        class_name = CLASS_MAPPING[int(prediction[0])]
    except Exception as e:
        # Handle unexpected errors during prediction
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

    # Return successful response with prediction result
    return jsonify({
        'prediction': class_name,
        'input': {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
    })

# Retrain the model with existing dataset and evaluate the new accuracy
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists(DATA_PATH):
        try:
            X_train, X_val, y_train, y_val = load_data(data_path=DATA_PATH)
            model = train_model(X_train=X_train, y_train=y_train)
            save_model(model=model, model_path=MODEL_PATH)
            
            accuracy = accuracy_score(y_true=y_val, y_pred=model.predict(X_val))
            return jsonify({'message': 'Model retrained successfully', 'accuracy': str(accuracy)})
            
        except Exception as e:
            return jsonify({'error': f'An error occurred during retraining: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Dataset for retraining not found'}), 404

# Calculate and return the accuracy of the current saved model on a validation dataset
@app.route('/api/v1/accuracy', methods=['GET'])
def accuracy():
    try:
        X_train, X_val, y_train, y_val = load_data(data_path=DATA_PATH)
        model = load_model(model_path=MODEL_PATH)
        accuracy = accuracy_score(y_true=y_val, y_pred=model.predict(X_val))
        return jsonify({'accuracy': str(accuracy)})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred during retraining: {str(e)}'}), 500

# Webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    repo_path = '/home/elecomexp/api_iris_model'
    server_wsgi = '/var/www/elecomexp_pythonanywhere_com_wsgi.py'

    if request.is_json:
        subprocess.run(['git', '-C', repo_path, 'pull'], check=True)
        subprocess.run(['touch', server_wsgi], check=True)
        return jsonify({'message': 'Despliegue actualizado con éxito'}), 200
    else:
        return jsonify({'error': 'Solicitud no válida'}), 400

# # Webhook
# @app.route("/webhook", methods=["POST"])
# def webhook():
#     # Ruta al repositorio donde se realizará el pull
#     path_repo = "/home/Elecomexp/despliegue_git"
#     servidor_web = "/var/www/elecomexp_pythonanywhere_com_wsgi.py"

#     # Comprueba si la solicitud POST contiene datos JSON
#     if request.is_json:
#         payload = request.json
#         # Verifica si la carga útil (payload) contiene información sobre el repositorio
#         if "repository" in payload:
#             # Extrae el nombre del repositorio y la URL de clonación
#             repo_name = payload["repository"]["name"]
#             clone_url = payload["repository"]["clone_url"]

#             # Cambia al directorio del repositorio
#             try:
#                 os.chdir(path_repo)
#             except FileNotFoundError:
#                 return jsonify({"message": "El directorio del repositorio no existe!"}), 404

#             # Realiza un git pull en el repositorio, trick to automatically reload PythonAnywhere WebServer
#             try:
#                 subprocess.run(["git", "pull", clone_url], check=True)
#                 subprocess.run(["touch", servidor_web], check=True)  
#                 return jsonify(
#                     {"message": f"Se realizó un git pull en el repositorio {repo_name}"}
#                 ), 200
#             except subprocess.CalledProcessError:
#                 return jsonify({"message": f"Error al realizar git pull en el repositorio {repo_name}"}), 500
#         else:
#             return jsonify({"message": "No se encontró información sobre el repositorio en la carga útil (payload)"}), 400
#     else:
#         return jsonify({"message": "La solicitud no contiene datos JSON"}), 400


# Main
if __name__ == '__main__':
    app.run()

