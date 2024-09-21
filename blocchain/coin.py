import ccxt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import talib

# Initialiser l'échange Binance
exchange = ccxt.binance()

# Initialiser le modèle de régression logistique
logreg = LogisticRegression()

# Initialiser le StandardScaler
scaler = StandardScaler()

while True:
    # Charger les données de prix de Bitcoin en temps réel depuis l'échange Binance
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['volume'], inplace=True)

    # Ajouter les indicateurs techniques
    df['ma20'] = df['close'].rolling(20).mean()
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df.dropna(inplace=True)

    # Préparer les données
    X = df[['open', 'high', 'low', 'close', 'ma20', 'rsi']].values
    X_scaled = scaler.fit_transform(X)
    
    # Entraîner le modèle de régression logistique
    y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    logreg.fit(X_scaled[:-1], y[:-1])
    # Prédire si l'utilisateur devrait acheter ou non Bitcoin
    if len(X_scaled) > 0:
        y_pred = logreg.predict(X_scaled[-1].reshape(1, -1))
        if y_pred == 1:
            print('Acheter Bitcoin')
        else:
            print('Ne pas acheter Bitcoin')
            
    # Entraîner le modèle sur les 50 derniers points de données
    if len(X_scaled) >= 50:
        y = np.where(np.diff(df['close'].values) > 0, 1, 0)
        logreg.fit(X_scaled[:-1], y)


