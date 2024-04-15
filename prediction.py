import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

height, width = 250, 250
path_model = './model/model.h5'
path_weights = './model/weights.weights.h5'
# cnn = tf.keras.models.load_model(path_model, custom_objects={'RMSprop': tf.keras.optimizers.RMSprop})

cnn = tf.keras.models.load_model(path_model)
cnn.load_weights(path_weights)
predicts = [
    "Aceite humectante", "Bolero liquido", "Brocha", "Cepillo",
    "Esponja lustradora", "Franela", "Grasa-crema", "Jabon de calabaza",
    "Brillo", "Tinta"
]

def predict(file):
    # Cargamos una imagen
    x = tf.keras.preprocessing.image.load_img(file,
                                              target_size=(height, width))
    # Convertimos la imagen en un arreglo
    x = tf.keras.preprocessing.image.img_to_array(x)
    # Agregando una dimension extra en el eje 0 del arreglo
    x = np.expand_dims(x, axis=0)
    # Hacer una predicción, contiene un arreglo en base al No de clases
    # [[0, 0, 0, 1]]
    data = cnn.predict(x)
    # Obtener la posición del arreglo con el valor más alto
    # En este caso retornara la posición de la clase con mas coincidencia
    prediction = np.argmax(data[0])
    print(f"Esto es prediction {prediction} <<=")
    # Hacemos validaciones en base al número de clases
    print(predicts[prediction])
    return predicts[prediction]


# Llamando la función y
# enviando la imagen a evaluar
# para una prediccion
# predict('filename.jpg')

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Clasificaor')
    root.geometry("550x300+300+150")
    root.resizable(width=True, height=True)

    def openfn():
        filename = filedialog.askopenfilename(title='open')
        return filename

    def open_img():
        x = openfn()
        img = Image.open(x)
        img = img.resize((250, 250), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.pack()
        clas = predict(x)
        classification = tk.Label(root, text=clas)
        classification.place(x=250, y=250, anchor='center')
        classification.pack()

    btn = tk.Button(root, text='open image', command=open_img).pack()
    root.mainloop()
