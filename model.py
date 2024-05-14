import itertools
import math
import random
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

def load_and_preprocess_data():
    pd.set_option('display.max_columns', None)

    # Load data
    data_urls = [
        ("1718", "https://football-data.co.uk/mmz4281/1718/all-euro-data-2017-2018.xlsx"),
        ("1819", "https://football-data.co.uk/mmz4281/1819/all-euro-data-2018-2019.xlsx"),
        ("1920", "https://football-data.co.uk/mmz4281/1920/all-euro-data-2019-2020.xlsx"),
        ("2021", "https://football-data.co.uk/mmz4281/2021/all-euro-data-2020-2021.xlsx"),
        ("2122", "https://football-data.co.uk/mmz4281/2122/all-euro-data-2021-2022.xlsx"),
        ("2223", "https://football-data.co.uk/mmz4281/2223/all-euro-data-2022-2023.xlsx"),
    ]
    
    raw_data = []
    for season, url in data_urls:
        print(season)
        raw_data.append(pd.read_excel(url, sheet_name='E0'))
    
    for data in raw_data[:2]:
        data.rename(columns = {
            'B365H':'AvgH', 'B365D':'AvgD', 'B365A':'AvgA',
            'BbAv>2.5':'AvgC>2.5', 'BbAv<2.5':'AvgC<2.5'
        }, inplace = True)

    columns_req = ['HomeTeam','AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD','AvgA', 'AvgC>2.5', 'AvgC<2.5']
    playing_statistics = [data[columns_req] for data in raw_data[:5]]

    playing_stat = pd.concat(playing_statistics, ignore_index=True)
    playing_stat.rename(columns={'AvgC>2.5': 'AvgMORE25', 'AvgC<2.5': 'AvgCLESS25'}, inplace=True)

    playing_stat.to_csv('playing_stat.csv', index=False)
    return playing_stat

def addNColumn1(pstat):
    cols = [
        "1GHR", "2GHR", "3GHR", "4GHR", "5GHR",
        "1GAR", "2GAR", "3GAR", "4GAR", "5GAR",
        "1HR", "2HR", "3HR",
        "1AR", "2AR", "3AR"
    ]
    for col in cols:
        pstat[col] = -1
    return pstat

def lastSixHomeOrAwayWDL(pstat, current):
    home_name = pstat.iloc[current]["HomeTeam"]
    away_name = pstat.iloc[current]["AwayTeam"]
    home_count, away_count = 0, 0
    home_specific_count, away_specific_count = 0, 0

    for idx in range(current - 1, -1, -1):
        if home_count == 5 and home_specific_count == 3 and away_count == 5 and away_specific_count == 3:
            break

        row = pstat.iloc[idx]
        if home_count < 5:
            if home_name == row["HomeTeam"] or home_name == row["AwayTeam"]:
                home_count += 1
                pstat.at[current, f"{home_count}GHR"] = str(row["FTR"])

        if home_specific_count < 3:
            if home_name == row["HomeTeam"]:
                home_specific_count += 1
                pstat.at[current, f"{home_specific_count}HR"] = str(row["FTR"])

        if away_count < 5:
            if away_name == row["AwayTeam"] or away_name == row["HomeTeam"]:
                away_count += 1
                pstat.at[current, f"{away_count}GAR"] = str(row["FTR"])

        if away_specific_count < 3:
            if away_name == row["AwayTeam"]:
                away_specific_count += 1
                pstat.at[current, f"{away_specific_count}AR"] = str(row["FTR"])

    return pstat

def calculate_goals_last_five(pstat):
    pstat['HomeGoalsScoredHome'] = 0
    pstat['HomeGoalsConcededHome'] = 0
    pstat['AwayGoalsScoredAway'] = 0
    pstat['AwayGoalsConcededAway'] = 0

    for index in range(len(pstat)):
        home_name = pstat.at[index, 'HomeTeam']
        away_name = pstat.at[index, 'AwayTeam']
        home_matches = pstat[(pstat['HomeTeam'] == home_name) & (pstat.index < index)].tail(5)
        away_matches = pstat[(pstat['AwayTeam'] == away_name) & (pstat.index < index)].tail(5)

        pstat.at[index, 'HomeGoalsScoredHome'] = home_matches['FTHG'].sum()
        pstat.at[index, 'HomeGoalsConcededHome'] = home_matches['FTAG'].sum()
        pstat.at[index, 'AwayGoalsScoredAway'] = away_matches['FTAG'].sum()
        pstat.at[index, 'AwayGoalsConcededAway'] = away_matches['FTHG'].sum()

    return pstat

def addNColumn3(pstat):
    cols = ["TPHH", "TPHA", "TPAH", "TPAA"]
    for col in cols:
        pstat[col] = -1
    return pstat

def lastFiveAveragePointsHomeAndAway(pstat, pstat_size):
    home_name = pstat.iloc[pstat_size - 1]["HomeTeam"]
    away_name = pstat.iloc[pstat_size - 1]["AwayTeam"]
    current = pstat_size - 1
    hh, ha, ah, aa = 0, 0, 0, 0
    tphh, tpha, tpah, tpaa = 0, 0, 0, 0

    i = pstat_size - 2
    while (hh < 5 or ha < 5 or ah < 5 or aa < 5) and i >= 0:
        match_home = pstat.iloc[i]["HomeTeam"]
        match_away = pstat.iloc[i]["AwayTeam"]
        ftr = pstat.iloc[i]["FTR"]

        if home_name == match_home and hh < 5:
            hh += 1
            if ftr == "H":
                tphh += 3
            elif ftr == "D":
                tphh += 1

        if home_name == match_away and ha < 5:
            ha += 1
            if ftr == "A":
                tpha += 3
            elif ftr == "D":
                tpha += 1

        if away_name == match_home and ah < 5:
            ah += 1
            if ftr == "H":
                tpah += 3
            elif ftr == "D":
                tpah += 1

        if away_name == match_away and aa < 5:
            aa += 1
            if ftr == "A":
                tpaa += 3
            elif ftr == "D":
                tpaa += 1

        i -= 1

    aphh = tphh / max(hh, 1)
    apha = tpha / max(ha, 1)
    apah = tpah / max(ah, 1)
    apaa = tpaa / max(aa, 1)

    pstat.at[current, "APHH"] = aphh
    pstat.at[current, "APHA"] = apha
    pstat.at[current, "APAH"] = apah
    pstat.at[current, "APAA"] = apaa

    return pstat

def process_data(file_name):
    x = pd.read_csv(file_name)

    x['1HR'] = np.where((x['1HR']=="H"), 1, x['1HR'])
    x['1HR'] = np.where((x['1HR']=="D"), 0, x['1HR'])
    x['1HR'] = np.where((x['1HR']=="A"), 2, x['1HR'])
    x['1HR'] = np.where((x['1HR']=="N"), -1, x['1HR'])

    cols = ['1HR', '2HR', '3HR', '1AR', '2AR', '3AR', '1GHR', '2GHR', '3GHR', '4GHR', '5GHR', '1GAR', '2GAR', '3GAR', '4GAR', '5GAR']
    for col in cols:
        x[col] = np.where((x[col]=="H"), 1, x[col])
        x[col] = np.where((x[col]=="D"), 0, x[col])
        x[col] = np.where((x[col]=="A"), 2, x[col])
        x[col] = np.where((x[col]=="N"), -1, x[col])

    x['AvgH'].fillna(2.5, inplace = True)
    x['AvgD'].fillna(5, inplace = True)
    x['AvgA'].fillna(2.5, inplace = True)
    x['AvgMORE25'].fillna(2, inplace = True)
    x['AvgCLESS25'].fillna(2, inplace = True)

    columns_to_exclude = ['HomeTeam', 'AwayTeam', 'FTR']
    columns_to_convert = [col for col in x.columns if col not in columns_to_exclude]
    x[columns_to_convert] = x[columns_to_convert].astype(float)
    x.replace(-1, np.nan, inplace=True)

    print("NaN values present before filling NaNs:")
    print(x.isnull().sum())

    x.fillna(-1, inplace=True)
    x.reset_index(drop=True, inplace=True)

    if x.empty:
        print("Error: DataFrame is empty after filling NaNs.")
        return None

    x.to_csv('output.csv', index=False)
    return x

def feature_selection(x):
    x_pre = x.drop(columns=[ 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])
    le = LabelEncoder()
    x_pre['FTR'] = le.fit_transform(x_pre['FTR'])

    X = x_pre.drop(columns=['FTR'])
    y = x_pre['FTR']

    if X.empty or y.empty:
        print("Error: X or y is empty after preprocessing.")
        return None

    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_

    feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)

    x_clean = x_pre.drop(columns=[ '1AR', '2HR', '3HR', '2AR', '3AR', '1HR', '1GHR', '4GAR', '5GAR', '2GAR', '3GHR', '1GAR', '2GHR', '4GHR', '3GAR', '5GHR'])
    return x_clean

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results[name] = {
            'Accuracy': accuracy,
            'Classification Report': report
        }
        predictions[name] = y_pred

    return results, predictions

def grid_search_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {}
        }
    }

    results = {}
    predictions = {}
    for name, config in models.items():
        model = config['model']
        params = config['params']
        clf = GridSearchCV(model, params, cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results[name] = {
            'Best Model': best_model,
            'Accuracy': accuracy,
            'Classification Report': report
        }
        predictions[name] = y_pred

    return results

def predict_last_20_games(last_20_games, results, scaler):
    last_20_games_features = last_20_games.drop(columns='FTR')
    last_20_games_scaled = scaler.transform(last_20_games_features)
    last_20_predictions = {}

    for name, result in results.items():
        model = result['Best Model']
        last_20_pred = model.predict(last_20_games_scaled)
        last_20_predictions[name] = last_20_pred

    last_20_results = pd.DataFrame({
        'Index': last_20_games.index,
        'FTR': last_20_games['FTR']
    })

    for name, preds in last_20_predictions.items():
        last_20_results[f'PredictedFTR_{name}'] = preds

    return last_20_results

def main():
    # Load and preprocess data
    playing_stat = load_and_preprocess_data()

    # Feature engineering
    playing_stat = addNColumn1(playing_stat)

    for i in range(len(playing_stat) - 1, 0, -1):
        playing_stat = lastSixHomeOrAwayWDL(playing_stat, i)
        if i % 100 == 0:
            print(i)

    playing_stat = calculate_goals_last_five(playing_stat)
    playing_stat = addNColumn3(playing_stat)

    i = len(playing_stat)
    while i > 1:
        playing_stat = lastFiveAveragePointsHomeAndAway(playing_stat, i)
        if i % 100 == 0:
            print(i)
        i -= 1

    # Save the processed data to playing_stat.csv
    playing_stat.to_csv('playing_stat.csv', index=False)

    # Process data from playing_stat.csv
    x = process_data('playing_stat.csv')
    if x is None or x.empty:
        print("Error: DataFrame is empty after processing.")
        return

    # Feature selection
    x_clean = feature_selection(x)
    if x_clean is None or x_clean.empty:
        print("Error: DataFrame is empty after feature selection.")
        return

    # Split dataset
    last_20_games = x_clean.tail(20)
    x_clean = x_clean.iloc[:-20]
    X = x_clean.drop(columns='FTR')
    y = x_clean['FTR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate models
    results, predictions = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    for name, result in results.items():
        print(f"Model: {name}")
        print(f"Accuracy: {result['Accuracy']}")
        print("Classification Report:")
        print(result['Classification Report'])
        print("\n" + "="*80 + "\n")

    # Grid search models
    grid_results = grid_search_models(X_train_scaled, X_test_scaled, y_train, y_test)
    for name, result in grid_results.items():
        print(f"Model: {name}")
        print(f"Best Model: {result['Best Model']}")
        print(f"Accuracy: {result['Accuracy']}")
        print("Classification Report:")
        print(result['Classification Report'])
        print("\n" + "="*80 + "\n")

    # Predict last 20 games
    last_20_results = predict_last_20_games(last_20_games, grid_results, scaler)
    print(last_20_results)

if __name__ == "__main__":
    main()
