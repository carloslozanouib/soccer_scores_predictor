#model_generator.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pickle
from config import LEAGUES

# Function to process each league
def process_league(league_id, league_name):
    raw_data_files = [
        f"static/data/1718/all-euro-data-2017-2018.xlsx",
        f"static/data/1819/all-euro-data-2018-2019.xlsx",
        f"static/data/1920/all-euro-data-2019-2020.xlsx",
        f"static/data/2021/all-euro-data-2020-2021.xlsx",
        f"static/data/2122/all-euro-data-2021-2022.xlsx",
        f"static/data/2223/all-euro-data-2022-2023.xlsx",
        f"static/data/2324/all-euro-data-2023-2024.xlsx",
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
    columns_req = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25']

    # Add missing columns with 0 values if they are not present in the dataframe
    for df in data_frames:
        for col in columns_req:
            if col not in df.columns:
                df[col] = 0
    
    # Remove duplicate columns, keeping only the first occurrence
    for i in range(len(data_frames)):
        data_frames[i] = data_frames[i].loc[:, ~data_frames[i].columns.duplicated()]
    
    # Select required columns and ensure consistency
    playing_statistics = [df[columns_req].copy() for df in data_frames]

    # Concatenate data
    updated_playing_stat = pd.concat(playing_statistics, ignore_index=True)

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

# Function to process and train each league
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
    print("\n")
    
    # Save the model with Pickle
    model_filepath = f'static/models/{league_id}/{league_id}_model.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

# Loop through each league and process
for league_id, league_name in LEAGUES.items():
    process_league(league_id, league_name)
    process_and_train_league(league_id)
