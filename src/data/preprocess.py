import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_images_and_labels(image_dir='data/raw'):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
            label = filename.split('_')[0]
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    images, labels = load_images_and_labels()
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)
