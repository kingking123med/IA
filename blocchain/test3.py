import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Téléchargement des données de l'action
df = yf.download('AAPL', start='2010-01-01', end='2022-02-14')

# Calcul des indicateurs techniques
df['ma20'] = df['Close'].rolling(window=20).mean()
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))
df.fillna(df.mean(), inplace=True)

# Séparation des données en ensembles d'entraînement et de test
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Création de l'ensemble d'entraînement
X_train = train_df[['Open', 'High', 'Low', 'Close', 'ma20', 'rsi']].values
y_train = np.where(train_df['Close'].shift(-1) > train_df['Close'], 1, -1)

# Création du classificateur RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du classificateur
rf.fit(X_train, y_train)

# Préparation des données pour la prédiction
last_df = df.tail(1)
last_df.drop(['Adj Close'], axis=1, inplace=True)
last_df.columns = ['open', 'high', 'low', 'close', 'volume', 'ma20', 'rsi']
last_df = last_df[['open', 'high', 'low', 'close', 'ma20', 'rsi']]
scaler = StandardScaler()
last_X_scaled = scaler.fit_transform(last_df)

# Prédiction
y_pred = rf.predict(last_X_scaled.reshape(1, -1))
print('Prédiction:', y_pred[0])

# Décision
if y_pred == 1:
    print('Acheter')
else:
    print('Vendre')
