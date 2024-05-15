import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB

# Funzione per caricare e preprocessare i dati
def load_and_preprocess_data():
    # Carica i dati dai file Excel
    raw_data_files = [
        "https://football-data.co.uk/mmz4281/1718/all-euro-data-2017-2018.xlsx",
        "https://football-data.co.uk/mmz4281/1819/all-euro-data-2018-2019.xlsx",
        "https://football-data.co.uk/mmz4281/1920/all-euro-data-2019-2020.xlsx",
        "https://football-data.co.uk/mmz4281/2021/all-euro-data-2020-2021.xlsx",
        "https://football-data.co.uk/mmz4281/2122/all-euro-data-2021-2022.xlsx",
        "https://football-data.co.uk/mmz4281/2223/all-euro-data-2022-2023.xlsx",
        "https://football-data.co.uk/mmz4281/2324/all-euro-data-2023-2024.xlsx"
    ]
    
    data_frames = []
    for file in raw_data_files:
        df = pd.read_excel(file, sheet_name='E0')
        data_frames.append(df)
    
    # Rinomina le colonne per coerenza
    for df in data_frames[:2]:
        df.rename(columns={'B365H':'AvgH', 'B365D':'AvgD', 'B365A':'AvgA', 'BbAv>2.5':'AvgMORE25', 'BbAv<2.5':'AvgCLESS25'}, inplace=True)
    
    # Aggiungi le colonne mancanti se necessario
    for df in data_frames:
        if 'AvgMORE25' not in df.columns:
            df['AvgMORE25'] = np.nan
        if 'AvgCLESS25' not in df.columns:
            df['AvgCLESS25'] = np.nan
    
    # Colonne richieste
    columns_req = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'AvgH', 'AvgD', 'AvgA', 'AvgMORE25', 'AvgCLESS25']
    
    # Seleziona le colonne richieste
    playing_statistics = [df[columns_req] for df in data_frames]
    
    # Concatena i DataFrame
    playing_stat = pd.concat(playing_statistics, ignore_index=True)
    
    # Salva il file CSV
    playing_stat.to_csv('playing_stat.csv', index=False)
    
    return playing_stat

# Funzione per aggiungere nuove feature
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

# Funzione principale
def main():
    # Carica e preprocessa i dati
    df = load_and_preprocess_data()
    print(f"Dati caricati: {df.shape}")

    # Aggiungi nuove feature
    df = add_new_features(df)
    print(f"Dati dopo l'aggiunta delle nuove feature: {df.shape}")

    # Converti i valori 'H', 'D', 'A' in numerici
    result_map = {'H': 1, 'D': 0, 'A': -1}
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['HomeTeam', 'AwayTeam', 'FTR']:
            df[col] = df[col].map(result_map)
    
    # Converti e pulisci i dati
    df = df.assign(
        AvgH = df['AvgH'].fillna(2.5),
        AvgD = df['AvgD'].fillna(5),
        AvgA = df['AvgA'].fillna(2.5),
        AvgMORE25 = df['AvgMORE25'].fillna(2),
        AvgCLESS25 = df['AvgCLESS25'].fillna(2)
    )

    columns_to_exclude = ['HomeTeam', 'AwayTeam', 'FTR']
    columns_to_convert = [col for col in df.columns if col not in columns_to_exclude]
    
    # Assicurati che tutte le colonne da convertire non contengano valori non numerici
    for col in columns_to_convert:
        print(f"Controllando la colonna: {col}")
        non_numeric = df[col].apply(lambda x: isinstance(x, str))
        if non_numeric.any():
            print(f"Valori non numerici trovati nella colonna {col}:")
            print(df[non_numeric][col].unique())
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dati dopo la pulizia: {df.shape}")

    # Seleziona le feature rilevanti
    df['FTR'] = LabelEncoder().fit_transform(df['FTR'])
    X = df.drop(columns=['FTR', 'HomeTeam', 'AwayTeam'])  # Escludiamo 'HomeTeam' e 'AwayTeam'
    y = df['FTR']

    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    top_features = feature_scores.head(20)['Feature'].tolist()
    X = X[top_features]
    print(f"Feature selezionate: {X.shape}")

    # Split del dataset in set di train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizzare le feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modello Naive Bayes
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Prevedere le FTR delle ultime 20 partite
    last_20_games = df.tail(20)
    df = df.iloc[:-20]

    # Usa le stesse feature selezionate
    last_20_games_features = last_20_games[top_features]
    last_20_games_scaled = scaler.transform(last_20_games_features)
    last_20_pred = model.predict(last_20_games_scaled)

    # Creare il dataset finale con index, FTR e PredictedFTR
    last_20_results = pd.DataFrame({
        'Index': last_20_games.index,
        'FTR': last_20_games['FTR']
    })

    last_20_results['Predicted_FTR'] = last_20_pred

    # Stampa i risultati
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("\n" + "="*80 + "\n")
    print(last_20_results)

if __name__ == "__main__":
    main()
