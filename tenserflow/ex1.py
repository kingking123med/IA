import tensorflow as tf
 

print(tf.__version__)
#The Fashion MNIST data is available directly in the tf.keras datasets API. You load it like this:
mnist = tf.keras.datasets.fashion_mnist
#Calling load_data on this object will give you two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#What does these values look like? Let's print a training image, and a training label to see...Experiment with different indices in the array. For example, also take a look at index 42...that's a a different boot than the one at index 0

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
#You'll notice that all of the values in the number are between 0 and 255. If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, a process called 'normalizing'...and fortunately in Python it's easy to normalize a list like this without looping. You do it like this:
training_images  = training_images / 255.0
test_images = test_images / 255.0
#Now you might be wondering why there are 2 sets...training and testing -- remember we spoke about this in the intro? The idea is to have 1 set of data for training, and then another set of data...that the model hasn't yet seen...to see how good it would be at classifying values. After all, when you're done, you're going to want to try it out with data that it hadn't previously seen!

#Let's now design the model. There's quite a few new concepts here, but don't worry, you'll get the hang of them.

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128, activation=tf.nn.relu),tf.keras.layers.Dense(10, activation=tf.nn.softmax)])  
                                    
                                    
#Séquentiel : qui définit une SÉQUENCE de couches dans le réseau de neurones

#Aplatir : vous vous souvenez plus tôt où nos images étaient un carré, lorsque vous les avez imprimées ? Aplatir prend simplement ce carré et le transforme en un ensemble unidimensionnel.

#Dense : Ajoute une couche de neurones

#Chaque couche de neurones a besoin d'une fonction d'activation pour leur dire quoi faire. Il y a beaucoup d'options, mais utilisez-les pour l'instant.

#Relu signifie en fait "Si X> 0 renvoie X, sinon renvoie 0" - donc ce qu'il fait, il ne transmet que les valeurs 0 ou plus à la couche suivante du réseau.

#Softmax prend un ensemble de valeurs et sélectionne effectivement la plus grande, donc, par exemple, si la sortie de la dernière couche ressemble à [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], il enregistre vous empêche de le parcourir à la recherche de la plus grande valeur et de le transformer en [0,0,0,0,1,0,0,0,0] -- Le but est d'économiser beaucoup de codage !

#La prochaine chose à faire, maintenant que le modèle est défini, est de le construire. Vous faites cela en le compilant avec un optimiseur et une fonction de perte comme avant - puis vous l'entraînez en appelant * model.fit * en lui demandant d'adapter vos données d'entraînement à vos étiquettes d'entraînement - c'est-à-dire de lui faire comprendre la relation entre le les données d'entraînement et leurs étiquettes réelles, donc à l'avenir, si vous avez des données qui ressemblent aux données d'entraînement, alors il peut faire une prédiction de ce à quoi ces données ressembleraient.
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
             
             
model.fit(training_images, training_labels, epochs=5)
#Une fois l'entraînement terminé, vous devriez voir une valeur de précision à la fin de l'époque finale. Cela pourrait ressembler à quelque chose comme 0,9098. Cela vous indique que votre réseau de neurones est précis à environ 91 % dans la classification des données d'entraînement. C'est-à-dire qu'il a trouvé une correspondance de modèle entre l'image et les étiquettes qui a fonctionné 91% du temps. Pas génial, mais pas mal étant donné qu'il n'a été formé que pendant 5 époques et qu'il a été fait assez rapidement.

#Mais comment cela fonctionnerait-il avec des données invisibles ? C'est pourquoi nous avons les images de test. Nous pouvons appeler model.evaluate et transmettre les deux ensembles, et il rapportera la perte pour chacun. Essayons:
model.evaluate(test_images, test_labels)
