from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Charger le modèle enregistré
model = load_model('minist.h5')
print('Modèle chargé depuis mon_modele.h5')

# Route pour recevoir les requêtes POST
@app.route('/classify', methods=['POST'])
def classify():
    # Charger l'image depuis la requête POST
    image_file = request.files['image']
    img = image.load_img(image_file, target_size=(224, 224)) # Ajuster la taille de l'image selon les besoins de votre modèle
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

 

    # Appeler la fonction de prédiction du modèle sur l'image
    result = model.predict(img)

    # Formater la réponse et l'envoyer au frontend React
    response = {'result': result.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
