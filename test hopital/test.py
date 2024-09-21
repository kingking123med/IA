import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def processs(path):

    # Read input data from a CSV file
    data = pd.read_csv(path)
    # Extraction des noms complets des patients
    data['Nom Complet'] = data['Prénom'] + ' ' + data['Nom']

    # Création d'une matrice TF-IDF pour représenter les noms complets des patients
    tfidf = TfidfVectorizer()
    name_matrix = tfidf.fit_transform(data['Nom Complet'])

    # Calcul de la similarité cosinus entre tous les paires de noms complets
    similarity_matrix = cosine_similarity(name_matrix)

    # Seuil de similarité pour considérer deux enregistrements comme des doublons
    threshold = 0.8

    # Liste pour stocker les paires de doublons détectées
    duplicates = []

    # Parcours de la matrice de similarité pour trouver les doublons
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                duplicates.append((i, j))

    # Fusion des doublons en un seul enregistrement cohérent
    for pair in duplicates:
        index1, index2 = pair
        # Vérification si les noms sont inversés
        try:
            name1 = data.loc[index1, 'Nom Complet']
            name2 = data.loc[index2, 'Nom Complet']
        except KeyError:
            print(f"Error: Invalid index - {index1} or {index2}")
            continue

        if name1.split()[::-1] == name2.split():
            # Fusion des attributs autres que le nom complet
            merged_attributes = {}
            for attribute in data.columns:
                if attribute != 'Nom Complet':
                    # Convert the columns to strings
                    merged_attributes[attribute] = set(str(data.loc[index1, attribute]).split(','))
                    merged_attributes[attribute].update(str(data.loc[index2, attribute]).split(','))
                    data.loc[index1, attribute] = ','.join(merged_attributes[attribute])
            # Suppression de l'enregistrement fusionné
            data.drop(index2, inplace=True)

    # Création d'une DataFrame pour stocker les enregistrements fusionnés
    grouped_data = pd.DataFrame(columns=['ID', 'Prénom', 'Nom', 'Opération', 'Labo', 'Nom Complet'])

    # Grouper les enregistrements par nom complet et fusionner les attributs
    for name in data['Nom Complet'].unique():
        filtered_data = data[data['Nom Complet'] == name]
        ids = filtered_data.index.tolist()

        # Fusion des attributs pour le même nom complet
        merged_attributes = {}
        for attribute in data.columns:
            if attribute != 'Nom Complet':
                merged_attributes[attribute] = set(','.join(str(value) for value in filtered_data[attribute]).split(','))


        # Ajouter l'enregistrement fusionné à la DataFrame
        new_row = pd.DataFrame({'ID': [','.join(map(str, ids))],
                                'Prénom': [filtered_data['Prénom'].iloc[0]],
                                'Nom': [filtered_data['Nom'].iloc[0]],
                                'Opération': [','.join(merged_attributes['Opération'])],
                                'Labo': [','.join(merged_attributes['Labo'])],
                                'Nom Complet': [name]})
        grouped_data = pd.concat([grouped_data, new_row], ignore_index=True)

    # Affichage des attributs fusionnés pour chaque patient

    # Exportation des données fusionnées vers un fichier CSV
    grouped_data.to_csv('output.csv', index=False)
    # Read input data from a CSV file
    data = pd.read_csv('output.csv')

    # Concatenate full names
    data['Nom Complet'] = data['Prénom'].str.split(',').str[0] + ' ' + data['Nom'].str.split(',').str[0]

    # Generate new IDs for each record
    data['ID'] = range(len(data))

    # Display the updated dataframe
    print(data)
    data.to_csv('data.csv', index=False) 