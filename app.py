from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Carregar os modelos salvos
with open("regression_model.pkl", "rb") as f:
    regressor = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("classification_model.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)


@app.route("/predict/regression", methods=["POST"])
def predict_regression():
    data = request.get_json()  # Recebe os dados em JSON
    year = data["year"]
    renewable_energy = data["renewable_energy"]

    # Processar e fazer a previsão
    scaled_data = scaler.transform([[year, renewable_energy]])
    prediction = regressor.predict(scaled_data)

    return jsonify({"predicted_emissions": prediction[0]})


@app.route("/predict/classification", methods=["POST"])
def predict_classification():
    data = request.get_json()  # Recebe os dados em JSON
    policies_implemented = data["policies_implemented"]
    efficiency_level = data["efficiency_level"]

    # Prever a região
    prediction = classifier.predict([[policies_implemented, efficiency_level]])

    return jsonify({"predicted_region": prediction[0]})


@app.route("/predict/cluster", methods=["POST"])
def predict_cluster():
    data = request.get_json()  # Recebe os dados em JSON
    clean_technologies = data["clean_technologies"]
    co2_emissions = data["co2_emissions"]

    # Prever o cluster
    prediction = kmeans.predict([[clean_technologies, co2_emissions]])

    return jsonify({"predicted_cluster": prediction[0]})


if __name__ == "__main__":
    app.run(debug=True)
