# import numpy as np
# from captcha_model import create_model
# from preprocess import load_images_and_labels, preprocess_data
# from tensorflow.keras.utils import to_categorical


# def train_model():
#     images, labels = load_images_and_labels()
#     X_train, X_test, y_train, y_test = preprocess_data(images, labels)
    
#     y_train = [list(map(lambda x: ord(x) - ord('0') if x.isdigit() else ord(x) - ord('A') + 10, label)) for label in y_train]
#     y_test = [list(map(lambda x: ord(x) - ord('0') if x.isdigit() else ord(x) - ord('A') + 10, label)) for label in y_test]
    
#     y_train = np.array([to_categorical(label, num_classes=36) for label in y_train])
#     y_test = np.array([to_categorical(label, num_classes=36) for label in y_test])
    
#     model = create_model()
#     model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
#     model.save('captcha_solver_model.h5')

# if __name__ == "__main__":
#     train_model()
import numpy as np
from tensorflow.keras.utils import to_categorical
from captcha_model import create_model
from preprocess import load_images_and_labels, preprocess_data

def train_model():
    images, labels = load_images_and_labels()
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)

    # Convert labels to numerical values and one-hot encode
    y_train = [list(map(lambda x: ord(x) - ord('0') if x.isdigit() else ord(x) - ord('A') + 10, label)) for label in y_train]
    y_test = [list(map(lambda x: ord(x) - ord('0') if x.isdigit() else ord(x) - ord('A') + 10, label)) for label in y_test]

    y_train = np.array([to_categorical(label, num_classes=36) for label in y_train])
    y_test = np.array([to_categorical(label, num_classes=36) for label in y_test])

    # Reshape y_train and y_test to be compatible with the model output
    y_train = y_train.reshape(-1, 36 * 4)
    y_test = y_test.reshape(-1, 36 * 4)

    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    model.save('captcha_solver_model.h5')

if __name__ == "__main__":
    train_model()




# ********************************************************************************************************************************************************
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

        window.geometry("+300+300")

        # Pedir al usuario que ingrese el texto real del CAPTCHA
        real_text = simpledialog.askstring("Entrada", "Ingrese el texto real del CAPTCHA:", parent=window)
        window.destroy()

        if not real_text or len(real_text) != 5:
            print("El texto del CAPTCHA debe tener exactamente 5 caracteres.")
            stop_requested = True
            continue

        # Convertir el texto a etiquetas
        labels = [ord(char) - ord('0') if char.isdigit() else ord(char) - ord('A') + 10 for char in real_text]
        labels = to_categorical(labels, num_classes=36).flatten()

        X.append(image)
        y.append(labels)

        # Entrenar el modelo en batch
        if len(X) == batch_size:
            X = np.array(X)
            y = np.array(y)
            model.fit(X, y, epochs=1)
            X, y = [], []

    # Entrenar con los datos restantes si hay menos de batch_size
    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        model.fit(X, y, epochs=1)

    root.destroy()

    # Entrenar con los datos restantes si hay menos de batch_size
    if len(X) > 0 and not stop_requested:
        X = np.array(X)
        y = np.array(y)
        model.fit(X, y, epochs=1)

    root.destroy()

if __name__ == "__main__":
    # Actualiza la ruta a las imágenes de CAPTCHA
    image_dir = os.path.join(os.path.dirname(__file__), '../../data/raw/')  # Asegúrate de que esta ruta exista
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

    model = create_model()
    train_interactively(model, image_paths)

    model.save('captcha_solver_model_interactive.h5')


# def train_interactively(model, image_paths, batch_size=10):
#     X = []
#     y = []

#     def stop_training():
#         nonlocal stop_requested
#         stop_requested = True

#     stop_requested = False

#     root = tk.Tk()
#     root.geometry("+100+100")  # Ajusta la posición de la ventana principal
#     root.title("Entrenamiento de CAPTCHA")

#     # stop_button = tk.Button(root, text="Detener Entrenamiento", command=stop_training)
#     # stop_button.pack()

#     for i, image_path in enumerate(image_paths):
#         if stop_requested:
#             break

#         image = preprocess_image(image_path)
#         predicted_text = predict_text(model, image)

#         # Mostrar la imagen y el texto predicho en una ventana
#         window = tk.Toplevel(root)
#         window.title(f'Imagen {i + 1}')
#         window.geometry("+100+300")  # Ajusta la posición de la ventana (x, y)

#         img = Image.open(image_path)
#         img = img.resize((320, 120))  # Cambiar el tamaño para mostrarlo mejor en tkinter
#         img = ImageTk.PhotoImage(img)

#         panel = tk.Label(window, image=img)
#         panel.image = img
#         panel.pack()

#         # Mostrar el texto predicho
#         label = tk.Label(window, text=f'Texto Predicho: {predicted_text}')
#         label.pack()

#         # Pedir al usuario que ingrese el texto real del CAPTCHA
#         real_text = simpledialog.askstring("Entrada", "Ingrese el texto real del CAPTCHA:", parent=window)
#         window.destroy()

#         if not real_text or len(real_text) != 4:
#             print("El texto del CAPTCHA debe tener exactamente 4 caracteres.")
#             stop_requested = True
#             continue

#         # Convertir el texto a etiquetas
#         labels = [ord(char) - ord('0') if char.isdigit() else ord(char) - ord('A') + 10 for char in real_text]
#         labels = to_categorical(labels, num_classes=36).flatten()

#         X.append(image)
#         y.append(labels)

#         # Entrenar el modelo en batch
#         if len(X) == batch_size:
#             X = np.array(X)
#             y = np.array(y)
#             model.fit(X, y, epochs=1)
#             X, y = [], []

#         root.update_idletasks()
#         root.update()


# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import to_categorical
# from PIL import Image, ImageTk
# import tkinter as tk

# def create_model(input_shape=(60, 160, 1), num_classes=10):  # 10 classes for digits 0-9
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dense(num_classes * 5, activation='softmax')  # Adjust the output layer for 5 characters
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (160, 60))  # Adjust the size according to your CAPTCHA dimensions
#     image = np.expand_dims(image, axis=-1)  # Add a dimension for the channel
#     image = image / 255.0  # Normalize
#     return image

# def predict_text(model, image):
#     prediction = model.predict(np.expand_dims(image, axis=0))
#     predicted_labels = np.argmax(prediction.reshape(5, 10), axis=1)
#     predicted_text = ''.join([str(label) for label in predicted_labels])
#     return predicted_text

# def custom_ask_string(title, prompt, parent):
#     def on_ok():
#         nonlocal user_input
#         user_input = entry.get()
#         dialog.destroy()

#     user_input = None
#     dialog = tk.Toplevel(parent)
#     dialog.title(title)
#     dialog.geometry(f"+{parent.winfo_x()+50}+{parent.winfo_y()+150}")  # Adjust the position of the child window
#     label = tk.Label(dialog, text=prompt)
#     label.pack(padx=20, pady=10)
#     entry = tk.Entry(dialog)
#     entry.pack(padx=20, pady=10)
#     ok_button = tk.Button(dialog, text="OK", command=on_ok)
#     ok_button.pack(pady=10)
#     dialog.transient(parent)
#     dialog.grab_set()
#     parent.wait_window(dialog)
#     return user_input

# def train_interactively(model, image_paths, batch_size=10):
#     X = []
#     y = []

#     def stop_training():
#         nonlocal stop_requested
#         stop_requested = True

#     stop_requested = False

#     root = tk.Tk()
#     root.geometry("+100+100")  # Adjust the position of the main window
#     root.title("Entrenamiento de CAPTCHA")

#     # stop_button = tk.Button(root, text="Detener Entrenamiento", command=stop_training)
#     # stop_button.pack()

#     for i, image_path in enumerate(image_paths):
#         if stop_requested:
#             break

#         image = preprocess_image(image_path)
#         predicted_text = predict_text(model, image)

#         # Show the image and predicted text in a window
#         window = tk.Toplevel(root)
#         window.title(f'Imagen {i + 1}')
#         window.geometry("+100+300")  # Adjust the position of the window (x, y)

#         img = Image.open(image_path)
#         img = img.resize((320, 120))  # Resize to show better in tkinter
#         img = ImageTk.PhotoImage(img)

#         panel = tk.Label(window, image=img)
#         panel.image = img
#         panel.pack()

#         # Show the predicted text
#         label = tk.Label(window, text=f'Texto Predicho: {predicted_text}')
#         label.pack()

#         # Ask the user to enter the real CAPTCHA text
#         real_text = custom_ask_string("Entrada", "Ingrese el texto real del CAPTCHA:", window)
#         window.destroy()

#         if not real_text or len(real_text) != 5:
#             print("El texto del CAPTCHA debe tener exactamente 5 caracteres.")
#             stop_requested = True
#             continue

#         # Convert the text to labels
#         labels = [int(char) for char in real_text]
#         labels = to_categorical(labels, num_classes=10).flatten()

#         X.append(image)
#         y.append(labels)

#         # Train the model in batches
#         if len(X) == batch_size:
#             X = np.array(X)
#             y = np.array(y)
#             model.fit(X, y, epochs=1)
#             X, y = [], []

#         root.update_idletasks()
#         root.update()

#     # Train with the remaining data if there are fewer than batch_size
#     if len(X) > 0 and not stop_requested:
#         X = np.array(X)
#         y = np.array(y)
#         model.fit(X, y, epochs=1)

#     root.destroy()

# if __name__ == "__main__":
#     # Update the path to the CAPTCHA images
#     image_dir = os.path.join(os.path.dirname(__file__), '../../data/raw/')  # Make sure this path exists
#     image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

#     model = create_model()
#     train_interactively(model, image_paths)

#     model.save('captcha_solver_model_interactive.h5')


# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import to_categorical
# from PIL import Image, ImageTk
# import tkinter as tk
# from tkinter import simpledialog

# def create_model(input_shape=(60, 160, 1), num_classes=36):
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dense(num_classes * 4, activation='softmax')  # Ajustar la última capa a 4 caracteres, cada uno con 36 clases
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (160, 60))  # Ajusta el tamaño según el tamaño de tus CAPTCHAs
#     image = np.expand_dims(image, axis=-1)  # Añadir una dimensión para el canal
#     image = image / 255.0  # Normalizar
#     return image

# def predict_text(model, image):
#     prediction = model.predict(np.expand_dims(image, axis=0))
#     predicted_labels = np.argmax(prediction.reshape(4, 36), axis=1)
#     predicted_text = ''.join([chr(label + ord('0')) if label < 10 else chr(label - 10 + ord('A')) for label in predicted_labels])
#     return predicted_text

# def train_interactively(model, image_paths, batch_size=10):
#     X = []
#     y = []

#     root = tk.Tk()
#     root.withdraw()  # Oculta la ventana principal

#     for i, image_path in enumerate(image_paths):
#         image = preprocess_image(image_path)
#         predicted_text = predict_text(model, image)
        
#         # Mostrar la imagen y el texto predicho en una ventana
#         window = tk.Toplevel(root)
#         window.title(f'Imagen {i + 1}')
#         window.geometry("+100+300")  # Ajusta la posición de la ventana (x, y)
        
#         img = Image.open(image_path)
#         img = img.resize((320, 120))  # Cambiar el tamaño para mostrarlo mejor en tkinter
#         img = ImageTk.PhotoImage(img)
        
#         panel = tk.Label(window, image=img)
#         panel.image = img
#         panel.pack()

#         # Mostrar el texto predicho
#         label = tk.Label(window, text=f'Texto Predicho: {predicted_text}')
#         label.pack()
        
#         window.geometry("+300+300")

#         # Pedir al usuario que ingrese el texto real del CAPTCHA
#         real_text = simpledialog.askstring("Entrada", "Ingrese el texto real del CAPTCHA:", parent=window)
#         window.destroy()

#         if not real_text or len(real_text) != 4:
#             print("El texto del CAPTCHA debe tener exactamente 4 caracteres.")
#             continue

#         # Convertir el texto a etiquetas
#         labels = [ord(char) - ord('0') if char.isdigit() else ord(char) - ord('A') + 10 for char in real_text]
#         labels = to_categorical(labels, num_classes=36).flatten()

#         X.append(image)
#         y.append(labels)

#         # Entrenar el modelo en batch
#         if len(X) == batch_size:
#             X = np.array(X)
#             y = np.array(y)
#             model.fit(X, y, epochs=1)
#             X, y = [], []

#     # Entrenar con los datos restantes si hay menos de batch_size
#     if len(X) > 0:
#         X = np.array(X)
#         y = np.array(y)
#         model.fit(X, y, epochs=1)

#     root.destroy()

# if __name__ == "__main__":
#     # Actualiza la ruta a las imágenes de CAPTCHA
#     image_dir = os.path.join(os.path.dirname(__file__), '../../data/raw/')  # Asegúrate de que esta ruta exista
#     image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]

#     model = create_model()
#     train_interactively(model, image_paths)

#     model.save('captcha_solver_model_interactive.h5')
