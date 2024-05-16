import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB

def create_single_match_dataframe(HomeTeam, AwayTeam, AvgH, AvgD, AvgA, AvgMORE25, AvgCLESS25):
    # Define the columns of the dataframe
    columns = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 
               'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25']
    
    # Create a dictionary with input values and default 0 for other columns
    match_data = {
        'HomeTeam': HomeTeam,
        'AwayTeam': AwayTeam,
        'FTR': 0,
        'FTHG': 0,
        'FTAG': 0,
        'HS': 0,
        'AS': 0,
        'HST': 0,
        'AST': 0,
        'HC': 0,
        'AC': 0,
        'HY': 0,
        'AY': 0,
        'HR': 0,
        'AR': 0,
        'AvgH': AvgH,
        'AvgD': AvgD,
        'AvgA': AvgA,
        'AvgMORE25': AvgMORE25,
        'AvgCLESS25': AvgCLESS25
    }
    
    # Create a DataFrame with a single row
    single_match_df = pd.DataFrame([match_data], columns=columns)
    
    return single_match_df

# Example usage (if you want to run this file independently)
if __name__ == "__main__":
    # Create new single match dataframe
    new_match_df = create_single_match_dataframe('Leeds', 'Norwich', 2.5, 3.2, 1.8, 2.1, 1.9)
    
    # Load existing data from Excel files
    raw_data_files = ["data/2324/all-euro-data-2023-2024.xlsx"]
    data_frames = []
    for file in raw_data_files:
        df = pd.read_excel(file, sheet_name="E1")
        data_frames.append(df)
    
    # Rename columns to match the required format
    for df in data_frames[:2]:
        df.rename(columns={'B365H':'AvgH', 'B365D':'AvgD', 'B365A':'AvgA', 'BbAv>2.5':'AvgMORE25', 'BbAv<2.5':'AvgCLESS25'}, inplace=True)
    
    # Ensure 'AvgMORE25' and 'AvgCLESS25' columns exist in all dataframes
    for df in data_frames:
        if 'AvgMORE25' not in df.columns:
            df['AvgMORE25'] = np.nan
        if 'AvgCLESS25' not in df.columns:
            df['AvgCLESS25'] = np.nan
    
    # Define the required columns
    columns_req = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25']
    
    # Select required columns from dataframes
    playing_statistics = [df[columns_req] for df in data_frames]
    
    # Concatenate the playing statistics
    playing_stat = pd.concat(playing_statistics, ignore_index=True)
    
    # Concatenate the new match dataframe with the existing data
    updated_playing_stat = pd.concat([playing_stat, new_match_df], ignore_index=True)
    
    # Display the updated dataframe
    updated_playing_stat.head()