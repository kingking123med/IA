import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Création du modèle CNN
model = Sequential()

# Première couche de convolution avec 32 filtres, 
# une taille de noyau de 3x3 et une fonction d'activation relu
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
# Couche de pooling avec une taille de noyau de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Deuxième couche de convolution avec 64 filtres, 
# une taille de noyau de 3x3 et une fonction d'activation relu
model.add(Conv2D(64, (3, 3), activation='relu'))
# Couche de pooling avec une taille de noyau de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Troisième couche de convolution avec 128 filtres, 
# une taille de noyau de 3x3 et une fonction d'activation relu
model.add(Conv2D(128, (3, 3), activation='relu'))
# Couche de pooling avec une taille de noyau de 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Aplatir les données
model.add(Flatten())
# Couche complètement connectée avec 512 neurones 
# et une fonction d'activation relu
model.add(Dense(512, activation='relu'))
# Couche de sortie avec une fonction d'activation sigmoïde (pour une classification binaire)
model.add(Dense(1, activation='sigmoid'))
# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Entraînement du modèle avec un générateur d'images
# pour la data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('/path/to/training_set', target_size=(256, 256), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('/path/to/test_set', target_size=(256, 256), batch_size=32, class_mode='binary')
model.fit(training_set, epochs=25, validation_data=test_set)
# Évaluation du modèle
scores = model.evaluate(test_set, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
import SimpleITK as sitk

# Charger l'image IRM
image_path = "path/to/image.nii.gz"
image = sitk.ReadImage(image_path)
# Normalisation de l'intensité des pixels
intensity_filter = sitk.IntensityWindowingImageFilter()
intensity_filter.SetWindowMinimum(0)
intensity_filter.SetWindowMaximum(200)
intensity_filter.SetOutputMinimum(0)
intensity_filter.SetOutputMaximum(1)
normalized_image = intensity_filter.Execute(image)
# Segmentation de la matière blanche et grise
threshold_filter = sitk.BinaryThresholdImageFilter()
threshold_filter.SetLowerThreshold(0.2)
threshold_filter.SetUpperThreshold(0.8)
threshold_filter.SetInsideValue(1)
threshold_filter.SetOutsideValue(0)
segmented_image = threshold_filter.Execute(normalized_image)
# Appliquer un masque pour obtenir la matière blanche uniquement
mask = sitk.ReadImage("path/to/mask.nii.gz")
masked_image = sitk.Mask(segmented_image, mask)
# Enregistrer l'image segmentée
sitk.WriteImage(masked_image, "path/to/segmented_image.nii.gz")
