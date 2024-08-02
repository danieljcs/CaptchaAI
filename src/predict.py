import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def predict_captcha(image, model_path='captcha_solver_model.h5'):
    model = load_model(model_path)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    pred = model.predict(np.array([image]))
    pred_text = ''.join([chr(np.argmax(p) + ord('0') if np.argmax(p) < 10 else np.argmax(p) + ord('A') - 10) for p in pred.reshape(4, 36)])
    return pred_text

if __name__ == "__main__":
    image_path = 'path_to_image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    prediction = predict_captcha(image)
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted: {prediction}')
    plt.show()
