import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pickle
import os

# Define leagues
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

def process_and_train_league(league_id):
    # Load file
    try:
        filepath = f"static/models/{league_id}/{league_id}_dataframe.csv"
        x = pd.read_csv(filepath, delimiter=",", header=0, index_col=None)
    except FileNotFoundError:
        print(f"File not found for league: {league_id}")
        return
    
    # Eliminate unnecessary columns
    studying_features = x.drop(columns=['HomeTeam', 'AwayTeam'])
    studying_features = studying_features.drop(columns=['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])
    studying_features = studying_features.drop(columns=['Last_Home_Red', 'AAHR', 'AHR', 'AAAR', 'AHAR', 'Last_Home_Yellow', 'Last_Away_Red', 'AHAYY', 'AAAYY', 'AHHY', 'Last_Away_Yellow'])
    
    # Create a copy of the DataFrame
    x = studying_features.copy()
    
    # Convert all columns to float
    for col in x.columns:
        if col != 'FTR':  # Exclude the target column
            x[col] = x[col].astype(float)
    
    # Separate features and target
    X = x.drop(columns=['FTR'])
    y = x['FTR']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model performance
    print(f"Classification report for league: {league_id}")
    print(classification_report(y_test, y_pred))
    
    # Save the model with Pickle
    model_filepath = f'static/models/{league_id}/{league_id}_model.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

# Loop through each league and process
for league_id in LEAGUES:
    process_and_train_league(league_id)