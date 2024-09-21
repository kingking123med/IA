from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

model = VGG16()
# Importation des 3 images suivantes : avion de guerre de type reaper, loup, ballon de football.
img1 = load_img('reaper.jpeg', target_size=(224, 224))
img2 = load_img('loup.jpeg', target_size=(224, 224))
img3 = load_img('ballon.jpg', target_size=(224, 224))
# Conversion image (matrice de pixels) en un numpy array
def preprocess(image) :
    image = img_to_array(image)

    # Redimensionnage 
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Preprocessing
    image = preprocess_input(image)
    
    return image
def pred_modele(image) :
    
    image = preprocess(image)
    # Prédiction
    y_pred = model.predict(image)

    # Conversion des probabilités en classe label
    label = decode_predictions(y_pred)
    
    # Affectation du label ayant la plus grande probabilité
    label = label[0][0]

 
    return ((label[1], label[2]*100))

img=[img1,img2,img3]

for i in range(3) :
    print("Prédiction image",i+1,":",pred_modele (img[i])[0], 'avec une probabilité de',round(pred_modele (img[i])[1],2),'%')