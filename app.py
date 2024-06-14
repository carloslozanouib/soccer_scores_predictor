#app.py

import json
import subprocess
import sys
from flask import Flask, request, jsonify, render_template, Response
from match_predictor import predict_multiple_matches, predict_match_result
from oracle import *
from config import LEAGUES, TEAMS
import logging
from tabulate import tabulate

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Cache future matches in memory
FUTURE_MATCHES = None

def load_future_matches():
    global FUTURE_MATCHES
    if FUTURE_MATCHES is None:
        try:
            with open('static/future_matches.json', 'r') as file:
                FUTURE_MATCHES = json.load(file)
            logger.info("Loaded future_matches.json into memory")
        except Exception as e:
            logger.error(f"Error loading future_matches.json: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html', leagues=LEAGUES)

@app.route('/get_teams/<league>')
def get_teams(league):
    teams = TEAMS.get(league, [])
    return jsonify(teams)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        results = []

        # Reverse the LEAGUES dictionary to map keys to names
        league_names = {v: k for k, v in LEAGUES.items()}

        # Flatten the JSON structure to match the required format
        for league_key, matches in FUTURE_MATCHES.items():
            league_name = LEAGUES.get(league_key, league_key)
            for match in matches:
                match_data = [
                    league_name,
                    match.get("Date"),
                    match.get("Time"),
                    match.get("HomeTeam"),
                    match.get("AwayTeam"),
                    match.get("AvgH"),
                    match.get("AvgD"),
                    match.get("AvgA"),
                    match.get("AvgMORE25"),
                    match.get("AvgCLESS25"),
                    match.get("Prediction")
                ]
                results.append(match_data)

        # Format the results as a table using tabulate
        headers = [
            "League", "Date", "Time", "HomeTeam", "AwayTeam",
            "AvgH", "AvgD", "AvgA", "AvgMORE25", "AvgCLESS25", "Prediction"
        ]
        formatted_table = tabulate(results, headers)

        return Response(formatted_table, mimetype='text/plain')
    except Exception as e:
        logger.error(f"Error in /predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500

def run_oracle_and_predict():
    try:
        logger.info("Running model_generator.py...")
        subprocess.run([sys.executable, 'model_generator.py'], check=True)
        logger.info("model_generator.py completed successfully.")

        logger.info("Running oracle.py...")
        subprocess.run([sys.executable, 'oracle.py'], check=True)
        logger.info("oracle.py completed successfully.")
        
        with open('static/future_matches.json', 'r') as file:
            future_matches = json.load(file)
        
        logger.debug('Loaded future_matches.json')

        for league in LEAGUES.keys():
            matches_of_interest = future_matches.get(league, [])
            if matches_of_interest:
                predictions = predict_multiple_matches(matches_of_interest, league)
                logger.debug(f"Predictions for league {league}: {predictions}")
                
                for match, prediction in zip(matches_of_interest, predictions[0]['Predictions']):
                    match['Prediction'] = prediction

        with open('static/future_matches.json', 'w') as file:
            json.dump(future_matches, file, indent=4)
        
        logger.info("Successfully updated future_matches.json with predictions.")
    except Exception as e:
        logger.error(f"Error in run_oracle_and_predict: {str(e)}")

if __name__ == '__main__':
    #run_oracle_and_predict()
    load_future_matches()
    app.run(host='0.0.0.0', port=8000)
