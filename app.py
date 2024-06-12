import json
import subprocess
import sys
from flask import Flask, request, jsonify, render_template
from match_predictor import predict_multiple_matches, predict_match_result
from oracle import *
from config import LEAGUES
from config import TEAMS
import logging

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html', leagues=LEAGUES)

@app.route('/get_teams/<league>')
def get_teams(league):
    teams = TEAMS.get(league, [])
    return jsonify(teams)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.debug(f"Received data for prediction: {data}")

        if not data or 'matches' not in data or not isinstance(data['matches'], list):
            logger.error("Invalid or missing 'matches' data")
            return jsonify({"error": "Invalid or missing 'matches' data"}), 400

        results = []

        for match in data['matches']:
            league = match.get('league')
            HomeTeam = match.get('HomeTeam')
            AwayTeam = match.get('AwayTeam')
            AvgH = match.get('AvgH')
            AvgD = match.get('AvgD')
            AvgA = match.get('AvgA')
            AvgMORE25 = match.get('AvgMORE25')
            AvgCLESS25 = match.get('AvgCLESS25')

            logger.debug(f"Match data: League={league}, HomeTeam={HomeTeam}, AwayTeam={AwayTeam}, AvgH={AvgH}, AvgD={AvgD}, AvgA={AvgA}, AvgMORE25={AvgMORE25}, AvgCLESS25={AvgCLESS25}")

            if not all([league, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25]):
                results.append({"error": "Incomplete match data"})
                continue

            try:
                prediction_result = predict_match_result(league, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25)
                logger.debug(f"Prediction result: {prediction_result}")

                if isinstance(prediction_result, dict) and 'Prediction' in prediction_result:
                    prediction = prediction_result['Prediction']
                else:
                    prediction = prediction_result

                prediction_text = {
                    "H": "Home Wins",
                    "A": "Away Wins",
                    "D": "Draw"
                }.get(prediction, "Data Error")

                results.append({
                    "HomeTeam": HomeTeam,
                    "AwayTeam": AwayTeam,
                    "Prediction": prediction_text
                })
            except Exception as e:
                logger.error(f"Error predicting match result: {str(e)}")
                results.append({"error": str(e)})

        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in /predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500

def run_oracle_and_predict():
    try:
        # Run model_generator.py using the virtual environment's Python interpreter
        logger.info("Running model_generator.py...")
        subprocess.run([sys.executable, 'model_generator.py'], check=True)
        logger.info("model_generator.py completed successfully.")

        # Run oracle.py using the virtual environment's Python interpreter
        logger.info("Running oracle.py...")
        subprocess.run([sys.executable, 'oracle.py'], check=True)
        logger.info("oracle.py completed successfully.")
        
        # Read future_matches.json
        with open('static/future_matches.json', 'r') as file:
            future_matches = json.load(file)
        
        # Predict results for each league
        for league, matches in future_matches.items():
            predictions = predict_multiple_matches(matches)
            future_matches[league] = predictions
        
        # Update future_matches.json with predictions
        with open('static/future_matches.json', 'w') as file:
            json.dump(future_matches, file, indent=4)
        
        logger.info("Successfully updated future_matches.json with predictions.")
    except Exception as e:
        logger.error(f"Error in run_oracle_and_predict: {str(e)}")

if __name__ == '__main__':
    # Run run_oracle_and_predict at app startup
    run_oracle_and_predict()
    app.run(host='0.0.0.0', port=8000)
