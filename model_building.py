import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB

def load_and_preprocess_data(league):
    raw_data_files = [
        f"data/1718/all-euro-data-2017-2018.xlsx",
        f"data/1819/all-euro-data-2018-2019.xlsx",
        f"data/1920/all-euro-data-2019-2020.xlsx",
        f"data/2021/all-euro-data-2020-2021.xlsx",
        f"data/2122/all-euro-data-2021-2022.xlsx",
        f"data/2223/all-euro-data-2022-2023.xlsx",
        f"data/2324/all-euro-data-2023-2024.xlsx",
    ]
    
    data_frames = []
    for file in raw_data_files:
        df = pd.read_excel(file, sheet_name=league)
        data_frames.append(df)
    
    for df in data_frames[:2]:
        df.rename(columns={'B365H':'AvgH', 'B365D':'AvgD', 'B365A':'AvgA', 'BbAv>2.5':'AvgMORE25', 'BbAv<2.5':'AvgCLESS25'}, inplace=True)
    
    for df in data_frames:
        if 'AvgMORE25' not in df.columns:
            df['AvgMORE25'] = np.nan
        if 'AvgCLESS25' not in df.columns:
            df['AvgCLESS25'] = np.nan
    
    columns_req = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25']
    
    playing_statistics = [df[columns_req] for df in data_frames]
    playing_stat = pd.concat(playing_statistics, ignore_index=True)
    
    return playing_stat

def add_new_features(df):
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
                    pstat.at[current, f"{home_count}GHR"] = row["FTR"]

            if home_specific_count < 3:
                if home_name == row["HomeTeam"]:
                    home_specific_count += 1
                    pstat.at[current, f"{home_specific_count}HR"] = row["FTR"]

            if away_count < 5:
                if away_name == row["AwayTeam"] or away_name == row["HomeTeam"]:
                    away_count += 1
                    pstat.at[current, f"{away_count}GAR"] = row["FTR"]

            if away_specific_count < 3:
                if away_name == row["AwayTeam"]:
                    away_specific_count += 1
                    pstat.at[current, f"{away_specific_count}AR"] = row["FTR"]

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

    for i in range(len(df) - 1, 0, -1):
        df = lastSixHomeOrAwayWDL(df, i)

    df = calculate_goals_last_five(df)
    return df

def train_and_predict(df, matches=None):
    result_map = {'H': 0, 'D': 1, 'A': 2}
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['HomeTeam', 'AwayTeam', 'FTR']:
            df[col] = df[col].map(result_map)
    
    df = df.assign(
        AvgH = df['AvgH'].fillna(2.5),
        AvgD = df['AvgD'].fillna(5),
        AvgA = df['AvgA'].fillna(2.5),
        AvgMORE25 = df['AvgMORE25'].fillna(2),
        AvgCLESS25 = df['AvgCLESS25'].fillna(2)
    )

    columns_to_exclude = ['HomeTeam', 'AwayTeam', 'FTR']
    columns_to_convert = [col for col in df.columns if col not in columns_to_exclude]
    
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['FTR'] = LabelEncoder().fit_transform(df['FTR'])
    X = df.drop(columns=['FTR', 'HomeTeam', 'AwayTeam'])
    y = df['FTR']

    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    top_features = feature_scores.head(20)['Feature'].tolist()
    X = X[top_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Debugging statements
    print(f"Training completed with accuracy: {accuracy}")
    print(f"Classification report: \n{report}")

    # Per salvare il modello
    with open('E1_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def main():
    league = 'E1'  # Example league
    data = load_and_preprocess_data(league)
    data_with_features = add_new_features(data)
    train_and_predict(data_with_features)

if __name__ == "__main__":
    main()
