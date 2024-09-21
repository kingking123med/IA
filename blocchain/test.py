import requests
import matplotlib.pyplot as plt

# Récupérer les données en temps réel de Bitcoin
while True:
    response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
    data = response.json()

    # Récupérer la valeur actuelle de Bitcoin en USD
    current_price = float(data['bpi']['USD']['rate'].replace(',', ''))

    # Ajouter la valeur actuelle au fichier de données
    with open('bitcoin_data.txt', 'a') as f:
        f.write(str(current_price) + '\n')

    # Lire le fichier de données
    with open('bitcoin_data.txt', 'r') as f:
        lines = f.readlines()
        x = [i for i in range(len(lines))]
        y = [float(line) for line in lines]

    # Afficher le graphe
    plt.style.use('dark_background')
    plt.plot(x, y)
    plt.title('Évolution de la courbe de Bitcoin')
    plt.xlabel('Temps (minutes)')
    plt.ylabel('Prix (USD)')
    plt.show()
