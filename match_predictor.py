#match_predictor.py

import pandas as pd
import numpy as np
import pickle
import logging
from config import num_last_matches

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def predict_match_result(matches, league_id):
    try:
        # Create a DataFrame from input matches if not already in DataFrame format
        if not isinstance(matches, pd.DataFrame):
            matches = pd.DataFrame(matches)
        
        matches['FTHG'] = np.nan
        matches['FTAG'] = np.nan

        csv_file_path = f"static/models/{league_id}/{league_id}_dataframe.csv"
        df = pd.read_csv(csv_file_path)
        
        columns_req = ['HomeTeam', 'AwayTeam', 'FTR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25', 'FTHG', 'FTAG']
        playing_stat = df[columns_req].copy()
        updated_playing_stat = pd.concat([playing_stat, matches], ignore_index=True)
        
        logger.debug(f"Updated dataframe: {updated_playing_stat.tail()}")

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

        i = len(updated_playing_stat)
        while i > 1:
            updated_playing_stat = lastFiveAveragePointsHomeAndAway(updated_playing_stat, i)
            if i % 100 == 0:
                logger.debug(i)
            i -= 1

        print(updated_playing_stat)

        updated_playing_stat = updated_playing_stat.drop(columns=['league', 'Date', 'Time'])

        x = updated_playing_stat.copy()
        x.replace(-1, pd.NA, inplace=True)
        x.dropna(inplace=True)
        x.reset_index(drop=True, inplace=True)

        studying_features = x.drop(columns=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        
        logger.debug(studying_features.columns)

        # Select the appropriate number of matches based on the league
        num_matches = num_last_matches.get(league_id, 10)
        last_row = studying_features.tail(num_matches)
        X_last = last_row.drop(columns=['FTR'])
        logger.debug(f"X_last: {X_last}")
        y_last = last_row['FTR']
        logger.debug(f"y_last: {y_last}")

        model_path = f'static/models/{league_id}/{league_id}_model.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        predictions = model.predict(X_last)
        logger.debug(f"Predictions: {predictions}")
        
        results = pd.DataFrame({
            'HomeTeam' : matches['HomeTeam'],
            'AwayTeam' : matches['AwayTeam'],
            'Predictions' : predictions
        })

        return results

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def predict_multiple_matches(matches, league):
    results = []
    try:
        predict = predict_match_result(matches, league)
        results.append(predict)
    except Exception as e:
        logger.error(f"Error predicting match result: {str(e)}")
        matches['error'] = str(e)
        results.append(matches)
    return results