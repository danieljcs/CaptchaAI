import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog

def create_model(input_shape=(60, 160, 1), num_classes=10):  # 10 classes for digits 0-9
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes * 5, activation='softmax')  # Adjust the output layer for 5 characters
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (160, 60))  # Adjust the size according to your CAPTCHA dimensions
    image = np.expand_dims(image, axis=-1)  # Add a dimension for the channel
    image = image / 255.0  # Normalize
    return image

def predict_text(model, image):
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_labels = np.argmax(prediction.reshape(5, 10), axis=1)
    predicted_text = ''.join([str(label) for label in predicted_labels])
    return predicted_text

def train_interactively(model, image_paths, batch_size=10):
    X = []
    y = []
    stop_requested = False
    root = tk.Tk()
    root.geometry("+100+400")  # Ajusta la posición de la ventana principal
    root.withdraw()  # Oculta la ventana principal

    for i, image_path in enumerate(image_paths):
        if stop_requested:
            break
        image = preprocess_image(image_path)
        predicted_text = predict_text(model, image)

        # Mostrar la imagen y el texto predicho en una ventana
        window = tk.Toplevel(root)
        window.title(f'Imagen {i + 1}')
        window.geometry("+100+400")  # Ajusta la posición de la ventana (x, y)

        img = Image.open(image_path)
        img = img.resize((320, 120))  # Cambiar el tamaño para mostrarlo mejor en tkinter
        img = ImageTk.PhotoImage(img)

        panel = tk.Label(window, image=img)
        panel.image = img
        panel.pack()

        # Mostrar el texto predicho
        label = tk.Label(window, text=f'Texto Predicho: {predicted_text}')
        label.pack(pady=20)


        def on_enter_press(event):
            on_ok()

        def onclose():
            nonlocal stop_requested
            print('CERRAR')
            stop_requested = True

        entry = tk.Entry(window)
        entry.bind("<Return>", on_enter_press)
        entry.pack(pady=10)
        entry.focus_set()

        def on_ok():
            nonlocal stop_requested
            entry.focus_set()
            real_text = entry.get()
            if not real_text or len(real_text) != 5:
                print("El texto del CAPTCHA debe tener exactamente 5 caracteres.")
                # stop_requested = True
            else:
                labels = [ord(char) - ord('0') if char.isdigit() else ord(char) - ord('A') + 10 for char in real_text]
                labels = to_categorical(labels, num_classes=36).flatten()
                X.append(image)
                y.append(labels)
                window.destroy()


        ok_button = tk.Button(window, text="OK", command=on_ok)
        ok_button.pack(pady=10)

        close_button = tk.Button(window, text="Cerrar", command=onclose)
        close_button.pack(pady=10)

        # window.transient(root)
        window.grab_set()
        root.wait_window(window)

        # Entrenar el modelo en batch
        if len(X) == batch_size:
            X = np.array(X)
            y = np.array(y)
            model.fit(X, y, epochs=1)
            X, y = [], []

    # Entrenar con los datos restantes si hay menos de batch_size
    if len(X) > 0 and not stop_requested:
        X = np.array(X)
        y = np.array(y)
        model.fit(X, y, epochs=1)

    root.destroy()

if __name__ == "__main__":
    # Actualiza la ruta a las imágenes de CAPTCHA
    image_dir = os.path.join(os.path.dirname(__file__), '../../data/raw/')  # Asegúrate de que esta ruta exista
    if not os.path.exists(image_dir):
        print(f"La ruta {image_dir} no existe. Por favor, verifica la ruta a las imágenes.")
    else:
        image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
        if not image_paths:
            print("No se encontraron imágenes .png en la ruta especificada.")
        else:
            model = create_model()
            train_interactively(model, image_paths)
            model.save('captcha_solver_model_interactive.h5')
