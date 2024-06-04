import pandas as pd
import numpy as np
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_single_match_dataframe(league_id, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25):
    # Define the columns of the dataframe
    columns = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 
               'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25']
    
    # Create a dictionary with input values and default np.nan for other columns
    match_data = {
        'HomeTeam': HomeTeam,
        'AwayTeam': AwayTeam,
        'FTR': np.nan,    # Assuming this column contains the result which is not known for the new match
        'FTHG': np.nan,   # Final Team Home Goals
        'FTAG': np.nan,   # Final Team Away Goals
        'HS': np.nan,     # Home Shots
        'AS': np.nan,     # Away Shots
        'HST': np.nan,    # Home Shots on Target
        'AST': np.nan,    # Away Shots on Target
        'HC': np.nan,     # Home Corners
        'AC': np.nan,     # Away Corners
        'HY': np.nan,     # Home Yellow Cards
        'AY': np.nan,     # Away Yellow Cards
        'HR': np.nan,     # Home Red Cards
        'AR': np.nan,     # Away Red Cards
        'AvgH': AvgH,     # Average Home Odds
        'AvgD': AvgD,     # Average Draw Odds
        'AvgA': AvgA,     # Average Away Odds
        'AvgMORE25': AvgMORE25,  # Average Over 2.5 Goals
        'AvgCLESS25': AvgCLESS25 # Average Under 2.5 Goals
    }
    
    # Create a DataFrame with a single row
    single_match_df = pd.DataFrame([match_data], columns=columns)
    
    return single_match_df

def predict_match_result(league_id, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25):
    try:
        # Create new single match dataframe
        new_match_df = create_single_match_dataframe(league_id, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25)
        
        # Load existing data from Excel files
        raw_data_files = ["static/data/2324/all-euro-data-2023-2024.xlsx"]
        data_frames = []
        for file in raw_data_files:
            df = pd.read_excel(file, sheet_name=league_id)
            data_frames.append(df)
        
        # Check if the source data contains the required columns
        for df in data_frames:
            logger.debug(f"Columns before renaming: {df.columns}")
        
        # Rename columns to match the required format and handle duplicate column names
        for df in data_frames:
            if 'B365H' in df.columns:
                df.rename(columns={'B365H': 'AvgH'}, inplace=True)
            if 'B365D' in df.columns:
                df.rename(columns={'B365D': 'AvgD'}, inplace=True)
            if 'B365A' in df.columns:
                df.rename(columns={'B365A': 'AvgA'}, inplace=True)
            if 'B365>2.5' in df.columns:
                df.rename(columns={'B365>2.5': 'AvgMORE25'}, inplace=True)
            else:
                logger.warning("Column 'B365>2.5' not found, 'AvgMORE25' will be NaN")
            if 'B365<2.5' in df.columns:
                df.rename(columns={'B365<2.5': 'AvgCLESS25'}, inplace=True)
            else:
                logger.warning("Column 'B365<2.5' not found, 'AvgCLESS25' will be NaN")
        
        # Ensure 'AvgMORE25' and 'AvgCLESS25' columns exist in all dataframes and check for NaNs
        for df in data_frames:
            if 'AvgMORE25' not in df.columns:
                df['AvgMORE25'] = np.nan
            if 'AvgCLESS25' not in df.columns:
                df['AvgCLESS25'] = np.nan
            logger.debug(f"AvgMORE25 has {df['AvgMORE25'].isna().sum()} NaNs")
            logger.debug(f"AvgCLESS25 has {df['AvgCLESS25'].isna().sum()} NaNs")
        
        # Remove duplicate columns, keeping only the first occurrence
        for i in range(len(data_frames)):
            data_frames[i] = data_frames[i].loc[:, ~data_frames[i].columns.duplicated()]

        # Define the required columns
        columns_req = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25']
        
        # Select required columns from dataframes and ensure column consistency
        playing_statistics = [df[columns_req].copy() for df in data_frames]
        
        # Concatenate the playing statistics and reset the index
        playing_stat = pd.concat(playing_statistics, ignore_index=True)

        # Concatenate the new match dataframe with the existing data and reset the index
        updated_playing_stat = pd.concat([playing_stat, new_match_df], ignore_index=True)
        
        # Display the updated dataframe
        logger.debug(f"Updated dataframe: {updated_playing_stat.tail()}")

        def calculate_goals_last_five(pstat):

            # Ensure the DataFrame has the necessary columns
            pstat['HomeGoalsScoredHome'] = 0
            pstat['HomeGoalsConcededHome'] = 0
            pstat['AwayGoalsScoredAway'] = 0
            pstat['AwayGoalsConcededAway'] = 0

            # Iterate for each match in the DataFrame
            for index in range(len(pstat)):
                home_name = pstat.at[index, 'HomeTeam']
                away_name = pstat.at[index, 'AwayTeam']

                # Find the last 5 matches prior to the current one for the home team
                home_matches = pstat[(pstat['HomeTeam'] == home_name) & (pstat.index < index)].tail(5)
                away_matches = pstat[(pstat['AwayTeam'] == away_name) & (pstat.index < index)].tail(5)

                # Calculate the goals scored and conceded
                pstat.at[index, 'HomeGoalsScoredHome'] = home_matches['FTHG'].sum()
                pstat.at[index, 'HomeGoalsConcededHome'] = home_matches['FTAG'].sum()
                pstat.at[index, 'AwayGoalsScoredAway'] = away_matches['FTAG'].sum()
                pstat.at[index, 'AwayGoalsConcededAway'] = away_matches['FTHG'].sum()

            return pstat

        r_x1 = calculate_goals_last_five(updated_playing_stat)

        def lastFiveAveragePointsHomeAndAway(pstat, pstat_size):
            home_name = pstat.iloc[pstat_size - 1]["HomeTeam"]
            away_name = pstat.iloc[pstat_size - 1]["AwayTeam"]
            current = pstat_size - 1
            hh, ha, ah, aa = 0, 0, 0, 0  # Match counters for each category
            # Total points accumulated in the last 5 valid matches for each category
            tphh, tpha, tpah, tpaa = 0, 0, 0, 0
            i = pstat_size - 2  # Start from the element immediately preceding
            while (hh < 5 or ha < 5 or ah < 5 or aa < 5) and i >= 0:
                match_home = pstat.iloc[i]["HomeTeam"]
                match_away = pstat.iloc[i]["AwayTeam"]
                ftr = pstat.iloc[i]["FTR"]
                # Home matches as home team
                if home_name == match_home and hh < 5:
                    hh += 1
                    if ftr == "H":
                        tphh += 3
                    elif ftr == "D":
                        tphh += 1
                # Home matches as away team
                if home_name == match_away and ha < 5:
                    ha += 1
                    if ftr == "A":
                        tpha += 3
                    elif ftr == "D":
                        tpha += 1
                # Away matches as home team
                if away_name == match_home and ah < 5:
                    ah += 1
                    if ftr == "H":
                        tpah += 3
                    elif ftr == "D":
                        tpah += 1
                # Away matches as away team
                if away_name == match_away and aa < 5:
                    aa += 1
                    if ftr == "A":
                        tpaa += 3
                    elif ftr == "D":
                        tpaa += 1
                i -= 1  # Decrement the index to move to the previous match
            # Calculate the average points for each category
            aphh = tphh / max(hh, 1)
            apha = tpha / max(ha, 1)
            apah = tpah / max(ah, 1)
            apaa = tpaa / max(aa, 1)
            # Assign the calculated values to the DataFrame
            pstat.at[current, "APHH"] = aphh
            pstat.at[current, "APHA"] = apha
            pstat.at[current, "APAH"] = apah
            pstat.at[current, "APAA"] = apaa
            return pstat

        i = len(r_x1)
        r_x2 = r_x1.copy()
        while(i > 1):
           r_x2 = lastFiveAveragePointsHomeAndAway(r_x1, i)
           if(i % 100 == 0):
             logger.debug(i)
           i -= 1
        logger.debug(r_x2.tail())

        def lastThreeAverageCardsHomeAndAway(pstat, pstat_size):
            home_name = pstat.iloc[pstat_size - 1]["HomeTeam"]
            away_name = pstat.iloc[pstat_size - 1]["AwayTeam"]
            current = pstat_size - 1
            hh, ha, ah, aa = 0, 0, 0, 0  # Match counters for each category
            # Total cards accumulated in the last 3 valid matches for each category
            thhy, thhr, thayy, thar = 0, 0, 0, 0  # Home team: yellow and red cards at home and away
            tahhy, tahhr, taayy, taar = 0, 0, 0, 0  # Away team: yellow and red cards at home and away
            i = pstat_size - 2  # Start from the element immediately preceding
            while (hh < 3 or ha < 3 or ah < 3 or aa < 3) and i >= 0:
                match_home = pstat.iloc[i]["HomeTeam"]
                match_away = pstat.iloc[i]["AwayTeam"]
                # Home matches as home team
                if home_name == match_home and hh < 3:
                    hh += 1
                    thhy += pstat.iloc[i]["HY"]
                    thhr += pstat.iloc[i]["HR"]
                # Home matches as away team
                if home_name == match_away and ha < 3:
                    ha += 1
                    thayy += pstat.iloc[i]["AY"]
                    thar += pstat.iloc[i]["AR"]
                # Away matches as home team
                if away_name == match_home and ah < 3:
                    ah += 1
                    tahhy += pstat.iloc[i]["HY"]
                    tahhr += pstat.iloc[i]["HR"]
                # Away matches as away team
                if away_name == match_away and aa < 3:
                    aa += 1
                    taayy += pstat.iloc[i]["AY"]
                    taar += pstat.iloc[i]["AR"]
                i -= 1  # Decrement the index to move to the previous match
            # Calculate the average cards for each category
            ahhy = thhy / max(hh, 1)
            ahr = thhr / max(hh, 1)
            ahayy = thayy / max(ha, 1)
            ahar = thar / max(ha, 1)
            aahhy = tahhy / max(ah, 1)
            aahr = tahhr / max(ah, 1)
            aaayy = taayy / max(aa, 1)
            aaar = taar / max(aa, 1)
            # Assign the calculated values to the DataFrame
            pstat.at[current, "AHHY"] = ahhy
            pstat.at[current, "AHR"] = ahr
            pstat.at[current, "AHAYY"] = ahayy
            pstat.at[current, "AHAR"] = ahar
            pstat.at[current, "AAHHY"] = aahhy
            pstat.at[current, "AAHR"] = aahr
            pstat.at[current, "AAAYY"] = aaayy
            pstat.at[current, "AAAR"] = aaar
            return pstat

        i = len(r_x2)
        r_x3 = r_x2.copy()
        while(i > 1):
           r_x3 = lastThreeAverageCardsHomeAndAway(r_x2, i)
           if(i % 100 == 0):
             logger.debug(i)
           i -= 1
        logger.debug(r_x3.tail())

        def lastFiveAverageGoalsHomeAndAway(pstat, pstat_size):
            home_name = pstat.iloc[pstat_size - 1]["HomeTeam"]
            away_name = pstat.iloc[pstat_size - 1]["AwayTeam"]
            current = pstat_size - 1
            hh, ha, ah, aa = 0, 0, 0, 0  # Match counters for each category
            # Total goals accumulated in the last 5 valid matches for each category
            thhg, thag, tahg, taag = 0, 0, 0, 0  # Home team: goals at home and away
            aahg, aaag, ahhg, ahag = 0, 0, 0, 0  # Away team: goals at home and away
            i = pstat_size - 2  # Start from the element immediately preceding
            while (hh < 5 or ha < 5 or ah < 5 or aa < 5) and i >= 0:
                match_home = pstat.iloc[i]["HomeTeam"]
                match_away = pstat.iloc[i]["AwayTeam"]
                # Home matches as home team
                if home_name == match_home and hh < 5:
                    hh += 1
                    thhg += pstat.iloc[i]["FTHG"]
                    thag += pstat.iloc[i]["FTAG"]
                # Home matches as away team
                if home_name == match_away and ha < 5:
                    ha += 1
                    tahg += pstat.iloc[i]["FTHG"]
                    taag += pstat.iloc[i]["FTAG"]
                # Away matches as home team
                if away_name == match_home and ah < 5:
                    ah += 1
                    aahg += pstat.iloc[i]["FTHG"]
                    aaag += pstat.iloc[i]["FTAG"]
                # Away matches as away team
                if away_name == match_away and aa < 5:
                    aa += 1
                    ahhg += pstat.iloc[i]["FTHG"]
                    ahag += pstat.iloc[i]["FTAG"]
                i -= 1  # Decrement the index to move to the previous match
            # Calculate the average goals for each category
            ahhg_avg = thhg / max(hh, 1)
            ahag_avg = thag / max(hh, 1)
            tahg_avg = tahg / max(ha, 1)
            taag_avg = taag / max(ha, 1)
            aahg_avg = aahg / max(ah, 1)
            aaag_avg = aaag / max(ah, 1)
            ahhg_away_avg = ahhg / max(aa, 1)
            ahag_away_avg = ahag / max(aa, 1)
            # Assign the calculated values to the DataFrame
            pstat.at[current, "AHHG"] = ahhg_avg
            pstat.at[current, "AHAG"] = ahag_avg
            pstat.at[current, "TAHG"] = tahg_avg
            pstat.at[current, "TAAG"] = taag_avg
            pstat.at[current, "AAHG"] = aahg_avg
            pstat.at[current, "AAAG"] = aaag_avg
            pstat.at[current, "AHHG_AWAY"] = ahhg_away_avg
            pstat.at[current, "AHAG_AWAY"] = ahag_away_avg
            return pstat

        i = len(r_x3)
        r_x4 = r_x3.copy()
        while(i > 1):
           r_x4 = lastFiveAverageGoalsHomeAndAway(r_x3, i)
           if(i % 100 == 0):
             logger.debug(i)
           i -= 1
        logger.debug(r_x4.tail())

        def lastMatchCards(pstat, pstat_size):
            home_name = pstat.iloc[pstat_size - 1]["HomeTeam"]
            away_name = pstat.iloc[pstat_size - 1]["AwayTeam"]
            current = pstat_size - 1
            last_home_yellow = last_home_red = last_away_yellow = last_away_red = None
            i = pstat_size - 2  # Start from the element immediately preceding
            while i >= 0:
                match_home = pstat.iloc[i]["HomeTeam"]
                match_away = pstat.iloc[i]["AwayTeam"]
                if last_home_yellow is None and last_home_red is None and (home_name == match_home or home_name == match_away):
                    if home_name == match_home:
                        last_home_yellow = pstat.iloc[i]["HY"]
                        last_home_red = pstat.iloc[i]["HR"]
                    else:
                        last_home_yellow = pstat.iloc[i]["AY"]
                        last_home_red = pstat.iloc[i]["AR"]
                if last_away_yellow is None and last_away_red is None and (away_name == match_home or away_name == match_away):
                    if away_name == match_home:
                        last_away_yellow = pstat.iloc[i]["HY"]
                        last_away_red = pstat.iloc[i]["HR"]
                    else:
                        last_away_yellow = pstat.iloc[i]["AY"]
                        last_away_red = pstat.iloc[i]["AR"]
                if (last_home_yellow is not None and last_home_red is not None) and (last_away_yellow is not None and last_away_red is not None):
                    break
                i -= 1  # Decrement the index to move to the previous match
            # Assign the calculated values to the DataFrame
            pstat.at[current, "Last_Home_Yellow"] = last_home_yellow
            pstat.at[current, "Last_Home_Red"] = last_home_red
            pstat.at[current, "Last_Away_Yellow"] = last_away_yellow
            pstat.at[current, "Last_Away_Red"] = last_away_red
            return pstat

        i = len(r_x4)
        r_x5 = r_x4.copy()
        while(i > 1):
           r_x5 = lastMatchCards(r_x4, i)
           if(i % 100 == 0):
             logger.debug(i)
           i -= 1
        logger.debug(r_x5.tail())

        x = r_x5.copy()

        # Replace all '-1' values with 'NaN'
        x.replace(-1, pd.NA, inplace=True)

        # Remove all rows that contain at least one 'NaN'
        # Separate the last row from the rest of the DataFrame
        last_row = x.tail(1)
        df_rest = x.iloc[:-1]

        # Remove all rows that contain at least one 'NaN' except the last one
        df_rest_clean = df_rest.dropna()

        # Recombine the cleaned DataFrame with the last row
        x = pd.concat([df_rest_clean, last_row], ignore_index=True)

        # Reset index
        x.reset_index(drop=True, inplace=True)

        studying_features = x.drop(columns=['HomeTeam', 'AwayTeam'])
        logger.debug(studying_features.columns)

        studying_features = studying_features.drop(columns=[ 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])
        logger.debug(studying_features)

        studying_features = studying_features.drop(columns=[ 'Last_Home_Red', 'AAHR', 'AHR', 'AAAR', 'AHAR', 'Last_Home_Yellow', 'Last_Away_Red', 'AHAYY', 'AAAYY', 'AHHY', 'Last_Away_Yellow'])
        logger.debug(studying_features.columns)

        # Select the last row of the DataFrame
        last_row = studying_features.tail(1)

        # Separate features (X) and target (y) for the last row
        X_last = last_row.drop(columns=['FTR'])
        y_last = last_row['FTR']

        logger.debug(f"X_last: {X_last}")
        logger.debug(f"y_last: {y_last}")

        # Load the model from the .pkl file
        model_path = f'static/models/{league_id}/{league_id}_model.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        predictions = model.predict(X_last)
        logger.debug(f"Predictions: {predictions}")
        
        # Return a dictionary with prediction and team information
        return {
            "HomeTeam": HomeTeam,
            "AwayTeam": AwayTeam,
            "Prediction": predictions[0]
        }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Replace with actual match details
    league_id = 'I1'
    HomeTeam = 'Fiorentina'
    AwayTeam = 'Napoli'
    AvgH = 2.25
    AvgD = 3.4
    AvgA = 3.10
    AvgMORE25 = 1.72
    AvgCLESS25 = 2.10
    result = predict_match_result(league_id, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25)
    print(result)