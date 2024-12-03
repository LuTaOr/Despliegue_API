import os
import pickle
from flask import Flask, jsonify, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Crear la aplicación Flask
app = Flask(__name__)
app.config["DEBUG"] = True

# Cargar o inicializar el modelo
MODEL_PATH = "iris_model.pkl"
DATA_PATH = "iris.csv"

if not os.path.exists(MODEL_PATH):
    # Si no existe el modelo, entrenarlo por primera vez
    data = pd.read_csv(DATA_PATH)
    X = data.iloc[:, :-1]
    y = data['species']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    pickle.dump(model, open(MODEL_PATH, "wb"))
else:
    # Cargar el modelo entrenado
    model = pickle.load(open(MODEL_PATH, "rb"))

# Landing Page
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Bienvenido a la API de predicción del modelo Iris",
        "endpoints": {
            "/api/v1/predict": "Proporciona predicciones basadas en las características de entrada (GET)",
            "/api/v1/retrain": "Reentrena el modelo con un nuevo dataset (GET)"
        }
    })

# Endpoint de predicción
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    try:
        sepal_length = float(request.args.get("sepal_length"))
        sepal_width = float(request.args.get("sepal_width"))
        petal_length = float(request.args.get("petal_length"))
        petal_width = float(request.args.get("petal_width"))
    except (TypeError, ValueError):
        return jsonify({"error": "Debe proporcionar todos los parámetros numéricos (sepal_length, sepal_width, petal_length, petal_width)"}), 400

    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return jsonify({"prediction": int(prediction[0])})

# Endpoint de reentrenamiento
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        X = data.iloc[:, :-1]
        y = data['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reentrenar el modelo
        new_model = LogisticRegression(max_iter=200)
        new_model.fit(X_train, y_train)

        # Evaluación
        accuracy = accuracy_score(y_test, new_model.predict(X_test))
        pickle.dump(new_model, open(MODEL_PATH, "wb"))

        return jsonify({
            "message": "Modelo reentrenado con éxito",
            "accuracy": accuracy
        })
    else:
        return jsonify({"error": "No se encontró el dataset para reentrenamiento"}), 404

# Iniciar la aplicación
if __name__ == "__main__":
    app.run()
