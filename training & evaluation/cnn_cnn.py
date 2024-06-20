import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from keras.utils import to_categorical

# Constants
IMAGE_SIZE = (100, 100)

# Function to load data from a directory
def load_data(directory):
    X = []
    y = []
    classes = sorted(os.listdir(directory))
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        images = os.listdir(class_dir)
        for image_name in images:
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
            X.append(image)
            y.append(class_id)
    return np.array(X), np.array(y)

# Main function
if __name__ == "__main__":
    # Define the directory containing the images
    data_dir = r'C:\Users\mishr\Field-Project-1st-year\preprocessed imagedata'

    # Load images and labels
    X, y = load_data(data_dir)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load first VGG16 model
    vgg_model_1 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Extract features using first VGG16 model
    X_train_features_1 = vgg_model_1.predict(X_train)
    X_val_features_1 = vgg_model_1.predict(X_val)

    # Flatten extracted features
    X_train_features_flat_1 = X_train_features_1.reshape(X_train_features_1.shape[0], -1)
    X_val_features_flat_1 = X_val_features_1.reshape(X_val_features_1.shape[0], -1)

    # Convert labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train)
    y_val_one_hot = to_categorical(y_val)

    # Define the second VGG16 model with sigmoid activation
    vgg_model_2 = Sequential([
        Dense(128, activation='sigmoid', input_shape=(X_train_features_flat_1.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='sigmoid'),
        Dropout(0.2),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    # Compile the second VGG16 model with categorical cross-entropy loss
    vgg_model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the second VGG16 model using features extracted from the first VGG16 model
    history = vgg_model_2.fit(X_train_features_flat_1, y_train_one_hot, epochs=1000, validation_data=(X_val_features_flat_1, y_val_one_hot), verbose=0)

    # Evaluate the model on the validation data
    val_loss, val_accuracy = vgg_model_2.evaluate(X_val_features_flat_1, y_val_one_hot)

    # Print the final validation accuracy and validation loss
    print("Final Validation Accuracy:", val_accuracy)
    print("Final Validation Loss:", val_loss)

    # Save the trained model
    model_save_path = r"C:\Users\mishr\Field-Project-1st-year\trained models\cnn_cnn_model_trained.h5"
    vgg_model_2.save(model_save_path)
    print("Trained model saved at:", model_save_path)

    # Plot training and validation accuracy and loss side by side
    plt.figure(figsize=(14, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()