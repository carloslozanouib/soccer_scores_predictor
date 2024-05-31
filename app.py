#app.py

from flask import Flask, request, jsonify, render_template
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
    data = request.get_json()
    results = []

    if 'matches' in data and isinstance(data['matches'], list):
        for match in data['matches']:
            league = match.get('league')
            HomeTeam = match.get('HomeTeam')
            AwayTeam = match.get('AwayTeam')
            AvgH = match.get('AvgH')
            AvgD = match.get('AvgD')
            AvgA = match.get('AvgA')
            AvgMORE25 = match.get('AvgMORE25')
            AvgCLESS25 = match.get('AvgCLESS25')

            if not all([league, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25]):
                results.append({"error": "Incomplete match data"})
            else:
                prediction_result = predict_match_result(league, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25)
                
                # Ensure the prediction is returned as a string
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
    else:
        return jsonify({"error": "No matches data found"}), 400

    # Ensure the HomeTeam is the first key in the JSON response
    for result in results:
        result = {key: result[key] for key in ["HomeTeam", "AwayTeam", "Prediction"]}

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)