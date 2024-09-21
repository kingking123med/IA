import requests
import matplotlib.pyplot as plt

# Récupérer les données en temps réel des principales cryptomonnaies
while True:
    response = requests.get('https://api.coingecko.com/api/v3/coins/markets?vs_currency=USD')
    data = response.json()

    # Dictionnaire pour stocker les données de chaque pièce
    coin_data = {}

    # Filtrer les données pour récupérer uniquement les pièces dont le prix est inférieur ou égal à 10 dollars
    for coin in data:
        if coin['current_price'] <= 10:
            coin_data[coin['symbol']] = {'prices': [], 'name': coin['name']}

    # Récupérer les prix actuels de chaque pièce et les stocker dans le dictionnaire
    for coin in coin_data:
        response = requests.get(f'https://api.coingecko.com/api/v3/coins/{coin.lower()}')
        data = response.json()
        current_price = data['market_data']['current_price']['usd']
        coin_data[coin]['prices'].append(current_price)

    # Créer un graphe de l'évolution des prix de chaque pièce
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    for coin in coin_data:
        plt.plot(coin_data[coin]['prices'], label=coin_data[coin]['name'])
    plt.title('Évolution des prix des cryptomonnaies dont la valeur est inférieure ou égale à 10 dollars')
    plt.xlabel('Temps (en heures)')
    plt.ylabel('Prix (en dollars)')
    plt.legend()
    plt.show()
