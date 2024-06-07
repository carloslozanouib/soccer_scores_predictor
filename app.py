#app.py

from flask import Flask, request, jsonify, render_template
from match_predictor import predict_match_result, predict_multiple_matches
from oracle import *
import logging
import schedule
import time
import threading
import subprocess
import json
import sys

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

TEAMS = {
    "E0": ["Burnley", "Arsenal", "Bournemouth", "Brighton", "Everton", "Sheffield United", "Newcastle", "Brentford", "Chelsea", "Man United", "Nott'm Forest", "Fulham", "Liverpool", "Wolves", "Tottenham", "Man City", "Aston Villa", "West Ham", "Crystal Palace", "Luton"],
    "E1": ["Sheffield Weds", "Blackburn", "Bristol City", "Middlesbrough", "Norwich", "Plymouth", "Stoke", "Swansea", "Watford", "Leicester", "Leeds", "Sunderland", "Coventry", "Birmingham", "Cardiff", "Huddersfield", "Hull", "Ipswich", "Millwall", "Preston", "Rotherham", "Southampton", "QPR", "West Brom"],
    "E2": ["Barnsley", "Blackpool", "Bolton", "Cambridge", "Carlisle", "Charlton", "Derby", "Northampton","Portsmouth", "Reading", "Shrewsbury", "Wycombe", "Bristol Rvs", "Burton", "Cheltenham", "Exeter","Fleetwood Town", "Leyton Orient", "Lincoln", "Oxford", "Peterboro", "Port Vale", "Stevenage", "Wigan"],
    "E3": ["Accrington", "Crawley Town", "Crewe", "Doncaster", "Forest Green", "Grimsby", "Morecambe", "Stockport", "Sutton", "Tranmere", "Wrexham", "AFC Wimbledon", "Barrow", "Bradford", "Gillingham", "Harrogate", "Mansfield", "Milton Keynes Dons", "Newport County", "Notts County", "Salford", "Swindon", "Walsall", "Colchester"],
    "EC": ["Aldershot", "Altrincham", "Barnet", "Chesterfield", "Gateshead", "Halifax", "Kidderminster", "Maidenhead", "Solihull", "Southend", "Wealdstone", "Rochdale", "Boreham Wood", "Dag and Red", "Dorking", "Eastleigh", "Ebbsfleet", "Fylde", "Hartlepool", "Oldham", "Oxford City", "Woking", "York", "Bromley"],
    "SC0": ["Celtic", "Dundee", "Livingston", "St Johnstone", "Kilmarnock", "Hibernian", "Rangers", "Ross County", "St Mirren", "Aberdeen", "Hearts", "Motherwell"],
    "SC1": ["Arbroath", "Dunfermline", "Inverness C", "Morton", "Partick", "Airdrie Utd", "Ayr", "Dundee United", "Queens Park", "Raith Rvs"],
    "SC2": ["Falkirk", "Hamilton", "Montrose", "Queen of Sth", "Stirling", "Alloa", "Annan Athletic", "Cove Rangers", "Edinburgh City", "Kelty Hearts"],
    "SC3": ["Bonnyrigg Rose", "East Fife", "Elgin", "Spartans", "Stenhousemuir", "Clyde", "Dumbarton", "Forfar", "Peterhead", "Stranraer"],
    "D1": ["Werder Bremen", "Augsburg", "Hoffenheim", "Leverkusen", "Stuttgart", "Wolfsburg", "Dortmund", "Union Berlin", "Ein Frankfurt", "RB Leipzig", "Bochum", "Darmstadt", "FC Koln", "Freiburg", "Heidenheim", "M'gladbach", "Mainz", "Bayern Munich"],
    "D2": ["Hamburg", "Hannover", "Kaiserslautern", "Osnabruck", "Wehen", "Fortuna Dusseldorf", "Braunschweig", "Greuther Furth", "Hansa Rostock", "Hertha", "Paderborn", "Elversberg", "Holstein Kiel", "St Pauli", "Schalke 04", "Karlsruhe", "Magdeburg", "Nurnberg"],
    "SP1": ["Almeria", "Sevilla", "Sociedad", "Las Palmas", "Ath Bilbao", "Celta", "Villarreal", "Getafe", "Cadiz", "Ath Madrid", "Mallorca", "Valencia", "Osasuna", "Girona", "Barcelona", "Betis", "Alaves", "Granada", "Real Madrid", "Vallecano"],
    "SP2": ["Amorebieta", "Valladolid", "Santander", "Zaragoza", "Elche", "Burgos", "Albacete", "Cartagena", "Leganes", "Mirandes", "Tenerife", "Andorra", "Eibar", "Espanol", "Levante", "Alcorcon", "Oviedo", "Sp Gijon", "Villarreal B", "Huesca", "Ferrol"],
    "I1": ["Empoli", "Frosinone", "Genoa", "Inter", "Roma", "Sassuolo", "Lecce", "Udinese", "Torino", "Bologna", "Monza", "Milan", "Verona", "Fiorentina", "Juventus", "Lazio", "Napoli", "Salernitana", "Cagliari", "Atalanta"],
    "I2": ["Bari", "Cosenza", "Cremonese", "Ternana", "Sudtirol", "Cittadella", "Parma", "Venezia", "Sampdoria", "Como", "FeralpiSalo", "Modena", "Catanzaro", "Ascoli", "Pisa", "Reggiana", "Palermo", "Brescia", "Spezia"],
    "F1": ["Nice", "Marseille", "Paris SG", "Brest", "Clermont", "Montpellier", "Nantes", "Rennes", "Strasbourg", "Metz", "Lyon", "Toulouse", "Lille", "Le Havre", "Lorient", "Reims", "Monaco", "Lens"],
    "F2": ["St Etienne", "Ajaccio", "Amiens", "Annecy", "Concarneau", "Dunkerque", "Laval", "Paris FC", "Valenciennes", "Pau FC", "Rodez", "Angers", "Auxerre", "Bastia", "Caen", "Grenoble", "Guingamp", "Quevilly Rouen", "Troyes", "Bordeaux"],
    "B1": ["St. Gilloise", "Eupen", "Charleroi", "RWD Molenbeek", "Antwerp", "Gent", "Club Brugge", "St Truiden","Standard", "Genk", "Cercle Brugge", "Oud-Heverlee Leuven", "Anderlecht", "Mechelen", "Westerlo", "Kortrijk","Lyon", "Paris FC", "Angers", "Amiens", "Annecy", "Concarneau", "Dunkerque", "Laval", "Paris FC", "Valenciennes","Pau FC", "Rodez", "Auxerre", "Bastia", "Caen", "Grenoble", "Guingamp", "Quevilly Rouen", "Troyes", "Bordeaux"],
    "N1": ["Volendam", "PSV Eindhoven", "Heerenveen", "Ajax", "Zwolle", "Nijmegen", "AZ Alkmaar", "Feyenoord", "Almere City", "Heracles", "Excelsior", "Vitesse", "For Sittard", "Go Ahead Eagles", "Utrecht", "Sparta Rotterdam", "Twente", "Waalwijk"],
    "P1": ["Sp Braga", "Gil Vicente", "Farense", "Sp Lisbon", "Rio Ave", "Estrela", "Arouca", "Moreirense", "Boavista", "Casa Pia", "Guimaraes", "Chaves", "Benfica", "Portimonense", "Estoril", "Porto", "Vizela", "Famalicao"],
    "T1": ["Trabzonspor", "Kasimpasa", "Konyaspor", "Kayserispor", "Pendikspor", "Sivasspor", "Ad. Demirspor", "Fenerbahce", "Alanyaspor", "Karagumruk", "Antalyaspor", "Istanbulspor", "Rizespor", "Galatasaray", "Hatayspor", "Buyuksehyr", "Besiktas", "Gaziantep", "Ankaragucu", "Samsunspor"],
    "G1": ["Volos NFC", "Giannina", "OFI Crete", "PAOK", "Olympiakos", "Panetolikos", "Asteras Tripolis", "Panathinaikos", "AEK", "Lamia", "Kifisia", "Panserraikos", "Aris", "Atromitos"]
    }


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
        # Esegui oracle.py usando l'interprete Python dell'ambiente virtuale
        subprocess.run([sys.executable, 'oracle.py'], check=True)
        
        # Leggi future_matches.json
        with open('static/future_matches.json', 'r') as file:
            future_matches = json.load(file)
        
        # Prevedi risultati per ogni lega
        for league, matches in future_matches.items():
            predictions = predict_multiple_matches(matches)
            future_matches[league] = predictions
        
        # Aggiorna future_matches.json con le previsioni
        with open('static/future_matches.json', 'w') as file:
            json.dump(future_matches, file, indent=4)
        
        logger.info("Successfully updated future_matches.json with predictions.")
    except Exception as e:
        logger.error(f"Error in run_oracle_and_predict: {str(e)}")

# Funzione per eseguire il task periodicamente
def run_schedule():
    # Attendi 24 ore prima di eseguire il primo task
    time.sleep(24 * 60 * 60)
    # Esegui il task ogni giorno
    schedule.every(1).days.do(run_oracle_and_predict)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Avvia il task scheduler in un thread separato
scheduler_thread = threading.Thread(target=run_schedule)
scheduler_thread.daemon = True
scheduler_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)