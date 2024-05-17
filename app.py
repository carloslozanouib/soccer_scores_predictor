from flask import Flask, request, jsonify
from match_predictor import predict_match_result

app = Flask(__name__)

LEAGUES = {
    "E0": "Premier League",
    "E1": "EFL Championship",
    "E2": "EFL League One",
    "E3": "EFL League Two",
    "EC": "National League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    "SP1": "La Liga",
    "SP2": "Segunda División",
    "I1": "Serie A",
    "I2": "Serie B",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "B1": "Belgian Pro League",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "T1": "Süper Lig",
    "G1": "Super League Greece",
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    league = data.get('league')
    match = data.get('match')  # Expecting a single match dictionary

    if league not in LEAGUES:
        return jsonify({"error": "Invalid league code"}), 400

    if not match:
        return jsonify({"error": "Match data is required"}), 400

    # Extract match details
    HomeTeam = match.get('HomeTeam')
    AwayTeam = match.get('AwayTeam')
    AvgH = match.get('AvgH')
    AvgD = match.get('AvgD')
    AvgA = match.get('AvgA')
    AvgMORE25 = match.get('AvgMORE25')
    AvgCLESS25 = match.get('AvgCLESS25')

    if not all([HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25]):
        return jsonify({"error": "Incomplete match data"}), 400

    # Perform the prediction using the function from 3.py
    prediction = predict_match_result(league, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25)

    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
