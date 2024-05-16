from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from model.prediction_model import load_and_preprocess_data, add_new_features, train_and_predict

app = Flask(__name__)

LEAGUES = {
    "E0" : "Premier League",
    "E1" : "EFL Championship",
    "E2" : "EFL League One",
    "EC" : "National League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1" : "Bundesliga",
    "D2" : "2. Bundesliga",
    "SP1": "La Liga",
    "SP2": "Segunda División",
    "I1" : "Serie A",
    "I2" : "Serie B",
    "F1" : "Ligue 1",
    "F2" : "Ligue 2",
    "B1" : "Belgian Pro League",
    "N1" : "Eredivisie",
    "P1" : "Primeira Liga",
    "T1" : "Süper Lig",
    "G1" : "Super League Greece",
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    league = data.get('league')
    matches = data.get('matches')
    
    if league not in LEAGUES:
        return jsonify({"error": "Invalid league code"}), 400

    # Load and preprocess data for the selected league
    df = load_and_preprocess_data(league)
    df = add_new_features(df)

    # Perform the prediction
    predictions, accuracy, report = train_and_predict(df, matches)
    
    # Format the predictions
    predictions_list = predictions.to_dict(orient='records')
    
    return jsonify({
        "predictions": predictions_list,
        "accuracy": accuracy,
        "report": report
    })

if __name__ == '__main__':
    app.run(debug=True)