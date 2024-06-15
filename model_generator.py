#model_generator.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pickle
import logging
from config import LEAGUES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_league(league_id, league_name):
    raw_data_files = [
        f"static/data/1718/all-euro-data-2017-2018.xlsx",
        f"static/data/1819/all-euro-data-2018-2019.xlsx",
        f"static/data/1920/all-euro-data-2019-2020.xlsx",
        f"static/data/2021/all-euro-data-2020-2021.xlsx",
        f"static/data/2122/all-euro-data-2021-2022.xlsx",
        f"static/data/2223/all-euro-data-2022-2023.xlsx",
        f"static/data/2324/all-euro-data-2023-2024-first.xlsx",
    ]

    data_frames = []
    for file in raw_data_files:
        df = pd.read_excel(file, sheet_name=league_id)
        data_frames.append(df)

    # Process columns and rename
    for df in data_frames:
        if 'B365H' in df.columns:
            df.rename(columns={'B365H': 'AvgH'}, inplace=True)
        if 'B365D' in df.columns:
            df.rename(columns={'B365D': 'AvgD'}, inplace=True)
        if 'B365A' in df.columns:
            df.rename(columns={'B365A': 'AvgA'}, inplace=True)
        if 'B365>2.5' in df.columns:
            df.rename(columns={'B365>2.5': 'AvgMORE25'}, inplace=True)
        elif 'BbAv>2.5' in df.columns:
            df.rename(columns={'BbAv>2.5': 'AvgMORE25'}, inplace=True)
        else:
            df['AvgMORE25'] = np.nan
        if 'B365<2.5' in df.columns:
            df.rename(columns={'B365<2.5': 'AvgCLESS25'}, inplace=True)
        elif 'BbAv<2.5' in df.columns:
            df.rename(columns={'BbAv<2.5': 'AvgCLESS25'}, inplace=True)
        else:
            df['AvgCLESS25'] = np.nan

    # Ensure columns exist
    for df in data_frames:
        if 'AvgMORE25' not in df.columns:
            df['AvgMORE25'] = np.nan
        if 'AvgCLESS25' not in df.columns:
            df['AvgCLESS25'] = np.nan

    # Remove duplicate columns
    for i in range(len(data_frames)):
        data_frames[i] = data_frames[i].loc[:, ~data_frames[i].columns.duplicated()]

    # Define required columns
    columns_req = ['HomeTeam', 'AwayTeam', 'FTR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25', 'FTHG', 'FTAG']

    # Select required columns and ensure consistency
    playing_statistics = [df[columns_req].copy() for df in data_frames]

    # Concatenate data
    updated_playing_stat = pd.concat(playing_statistics, ignore_index=True)

    print(updated_playing_stat.tail())

    def calculate_goals_last_five(pstat):
        if 'FTHG' not in pstat.columns or 'FTAG' not in pstat.columns:
            return pstat
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

    updated_playing_stat = calculate_goals_last_five(updated_playing_stat)

    def lastFiveAveragePointsHomeAndAway(pstat, pstat_size):
        home_name = pstat.iloc[pstat_size - 1]["HomeTeam"]
        away_name = pstat.iloc[pstat_size - 1]["AwayTeam"]
        current = pstat_size - 1
        hh, ha, ah, aa = 0, 0, 0, 0  # Counters for the last 5 matches
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

    i = len(updated_playing_stat)
    while i > 1:
        updated_playing_stat = lastFiveAveragePointsHomeAndAway(updated_playing_stat, i)
        if i % 100 == 0:
            print(i)
        i -= 1

    # Clean and save the dataframe
    x = updated_playing_stat.copy()
    x.replace(-1, pd.NA, inplace=True)
    x.dropna(inplace=True)
    x.reset_index(drop=True, inplace=True)

    # Ensure directory exists
    output_dir = f'static/models/{league_id}'
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    x.to_csv(f'{output_dir}/{league_id}_dataframe.csv', index=False)

    # Train and save model
    process_and_train_league(league_id)

def process_and_train_league(league_id):
    try:
        filepath = f'static/models/{league_id}/{league_id}_dataframe.csv'
        x = pd.read_csv(filepath, delimiter=",", header=0, index_col=None)
    except FileNotFoundError:
        print(f"File not found for league: {league_id}")
        return
    
    studying_features = x.drop(columns=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    
    x = studying_features.copy()
    
    for col in x.columns:
        if col != 'FTR':
            x[col] = x[col].astype(float)
    
    X = x.drop(columns=['FTR'])
    y = x['FTR']
    
    # Use cross-validation to better understand model performance
    model = GaussianNB()
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    logger.info(f"Cross-validation F1 scores for league {league_id}: {scores}")
    logger.info(f"Mean cross-validation F1 score for league {league_id}: {scores.mean()}")

    # Train the Naive Bayes model
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    logger.info(f"Classification report for league {league_id}")
    logger.info(classification_report(y, y_pred))
    
    # Save the model with Pickle
    model_filepath = f'static/models/{league_id}/{league_id}_model.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

# Loop through each league
for league_id, league_name in LEAGUES.items():
    process_league(league_id, league_name)