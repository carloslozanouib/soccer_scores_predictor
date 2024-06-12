import pandas as pd
import numpy as np
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_single_match_dataframe(HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25):
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

def preprocess_data(df):
    # Fill missing values in 'AvgMORE25' and 'AvgCLESS25' with the mean of the column
    if 'AvgMORE25' in df.columns and df['AvgMORE25'].isna().any():
        df['AvgMORE25'].fillna(df['AvgMORE25'].mean(), inplace=True)
    if 'AvgCLESS25' in df.columns and df['AvgCLESS25'].isna().any():
        df['AvgCLESS25'].fillna(df['AvgCLESS25'].mean(), inplace=True)
    
    # Fill any remaining NaN values with zero
    df.fillna(0, inplace=True)
    
    return df

def predict_match_result(league_id, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25):
    try:
        # Create new single match dataframe
        new_match_df = create_single_match_dataframe(HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25)
        
        # Preprocess the data
        new_match_df = preprocess_data(new_match_df)
        
        # Load the model from the .pkl file
        model_path = f'static/models/{league_id}/{league_id}_model.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Ensure that the columns match those expected by the model
        expected_columns = model.feature_names_in_
        new_match_df = new_match_df[expected_columns]
        
        # Make predictions using the model
        predictions = model.predict(new_match_df)
        
        # Return a dictionary with prediction and team information
        return {
            "HomeTeam": HomeTeam,
            "AwayTeam": AwayTeam,
            "Prediction": predictions[0]
        }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def predict_multiple_matches(matches):
    results = []
    for match in matches:
        try:
            prediction_result = predict_match_result(
                match['league'],
                match['HomeTeam'],
                match['AwayTeam'],
                match['AvgH'],
                match['AvgD'],
                match['AvgA'],
                match['AvgMORE25'],
                match['AvgCLESS25']
            )
            match.update(prediction_result)
            results.append(match)
        except Exception as e:
            logger.error(f"Error predicting match result: {str(e)}")
            match['error'] = str(e)
            results.append(match)
    return results

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
