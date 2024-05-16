import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pickle

# Load file (Prima riga ci sono le label e la prima colonna ha gli indici)
x = pd.read_csv("dataframe.csv", delimiter=",", header=0, index_col=None)

# Elimina le colonne non necessarie
studying_features = x.drop(columns=['HomeTeam', 'AwayTeam'])
print(studying_features.columns)
studying_features = studying_features.drop(columns=['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'])
print(studying_features)
studying_features = studying_features.drop(columns=['Last_Home_Red', 'AAHR', 'AHR', 'AAAR', 'AHAR', 'Last_Home_Yellow', 'Last_Away_Red', 'AHAYY', 'AAAYY', 'AHHY', 'Last_Away_Yellow'])
print(studying_features.columns)

# Crea una copia del DataFrame
x = studying_features.copy()

# Converti tutte le colonne a valori float
for col in x.columns:
    if col != 'FTR':  # Escludi la colonna target
        x[col] = x[col].astype(float)

# Separa le feature dal target
X = x.drop(columns=['FTR'])
y = x['FTR']

# Dividi i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestra il modello Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Fai previsioni sul set di test
y_pred = model.predict(X_test)

# Valuta le prestazioni del modello
print(classification_report(y_test, y_pred))

# Salva il modello con Pickle
with open('E1_model.pkl', 'wb') as file:
    pickle.dump(model, file)
