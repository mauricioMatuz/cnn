import sys
import os

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

#! Limpia cualquier sesión de TensorFlow previa para evitar conflictos
tf.keras.backend.clear_session()

#! Rutas a los directorios que contienen los conjuntos de datos de entrenamiento y validación
path_training = "./data/training"
path_validate = "./data/validate"

#! Hiperparámetros del modelo y del proceso de entrenamiento
epochs = 1
height, width = 250, 250  #! Tamaño de las imágenes de entrada
batch_size = 1  #! Tamaño del lote para el entrenamiento
filter_conv1 = 32  #! Número de filtros convolucionales en la primera capa
filter_conv2 = 64  #! Número de filtros convolucionales en la segunda capa
size_filter1 = (3, 3)  #! Tamaño del filtro para la primera capa convolucional
size_filter2 = (2, 2)  #! Tamaño del filtro para la segunda capa convolucional
size_pool = (2, 2)  #! Tamaño del pooling
classes = 10  #! Número de clases en la clasificación
lr = 0.0001  #! Tasa de aprendizaje del optimizador

#! Configuración del generador de datos de imágenes para el conjunto de entrenamiento
datagen_training = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  #! Normaliza los valores de píxeles a un rango de [0,1]
    shear_range=0.3,  #! Aplica cortes aleatorios a las imágenes
    zoom_range=0.3,  #! Aplica zoom aleatorio a las imágenes
    horizontal_flip=True)  #! Voltea aleatoriamente las imágenes horizontalmente

#! Genera el flujo de datos de imágenes para el conjunto de entrenamiento
img_training = datagen_training.flow_from_directory(
    path_training,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',  #! Problema de clasificación multiclase
    shuffle=False)  #! No baraja las imágenes (ya se han barajado previamente)

#! Configuración del generador de datos de imágenes para el conjunto de validación
datagen_validate = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)  #! Solo normaliza los valores de píxeles

#! Genera el flujo de datos de imágenes para el conjunto de validación
img_validate = datagen_validate.flow_from_directory(path_validate,
                                                    target_size=(height,
                                                                 width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

#! Define la arquitectura del modelo de red neuronal convolucional (CNN)
cnn = Sequential()
cnn.add(
    Conv2D(
        filter_conv1,
        size_filter1,
        padding='same',  #! Conserva el tamaño de la salida
        input_shape=(
            height, width, 3
        ),  #! Tamaño de las imágenes de entrada (altura, ancho, canales RGB)
        activation='relu')
)  #! Función de activación ReLU para la primera capa convolucional
cnn.add(MaxPooling2D(
    pool_size=size_pool))  #! Capa de pooling para reducir la dimensionalidad
cnn.add(Conv2D(filter_conv2, size_filter2, padding='same',
               activation='relu'))  #! Segunda capa convolucional
cnn.add(MaxPooling2D(pool_size=size_pool))  #! Capa de pooling
cnn.add(Flatten())  #! Aplana la salida para conectarla a una capa densa
cnn.add(Dense(
    256, activation='relu'))  #! Capa densa con 256 neuronas y activación ReLU
cnn.add(Dropout(0.5))  #! Dropout para evitar el sobreajuste
cnn.add(
    Dense(classes, activation='softmax')
)  #! Capa de salida con activación softmax para clasificación multiclase
cnn.summary()  #! Muestra un resumen de la arquitectura del modelo

#! Compila el modelo especificando la función de pérdida, el optimizador y las métricas de evaluación
cnn.compile(
    loss=
    'categorical_crossentropy',  #! Entropía cruzada categórica para problemas de clasificación multiclase
    optimizer='rmsprop',  #! Optimizador RMSprop para el ajuste de los pesos
    metrics=['accuracy'
             ])  #! Métrica de precisión para evaluar el rendimiento del modelo

#! Entrena el modelo en los datos de entrenamiento, utilizando los datos de validación para la validación
history = cnn.fit(img_training, epochs=epochs, validation_data=img_validate)

#! Crea un directorio para almacenar los modelos guardados si no existe
directory = './model/'
if not os.path.exists(directory):
    os.mkdir(directory)

#! Guarda el modelo y los pesos en archivos separados
cnn.save('./model/model.h5'
         )  #! Guarda la arquitectura del modelo y sus pesos en un archivo H5
cnn.save_weights('./model/weights.weights.h5'
                 )  #! Guarda únicamente los pesos del modelo en un archivo H5

#! Extrae los datos de la historia de entrenamiento para graficar las métricas de rendimiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#! Grafica la precisión del entrenamiento y validación a lo largo de las épocas
plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#! Grafica la pérdida del entrenamiento y validación a lo largo de las épocas
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#! Realiza predicciones en el conjunto de entrenamiento y muestra la matriz de confusión
predictions = cnn.predict(img_training)
pred_y = np.argmax(predictions, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(img_training.classes, pred_y)
print(cm)

#! Muestra la matriz de confusión como una tabla
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
plt.title('Confusion Matrix')
ax.table(cellText=cm,
         loc='center',
         colLabels=[
             "Aceite humectante", "Bolero liquido", "Brocha", "Cepillo",
             "Esponja lustradora", "Franela", "Grasa-crema",
             "Jabon de calabaza", "Brillo", "Tinta"
         ],
         rowLabels=[
             "Aceite humectante", "Bolero liquido", "Brocha", "Cepillo",
             "Esponja lustradora", "Franela", "Grasa-crema",
             "Jabon de calabaza", "Brillo", "Tinta"
         ])
fig.tight_layout()
plt.show()

#! Muestra el informe de clasificación
print('Classification Report')
cr = classification_report(img_training.classes,
                           pred_y,
                           target_names=[
                               "Aceite humectante", "Bolero liquido", "Brocha",
                               "Cepillo", "Esponja lustradora", "Franela",
                               "Grasa-crema", "Jabon de calabaza", "Brillo",
                               "Tinta"
                           ])
print(cr)
