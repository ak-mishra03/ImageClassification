import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import VGG16
import joblib

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

    # Train the ensemble classifier on CNN features
    ensemble_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    ensemble_classifier.fit(X_train_features_flat_1, y_train)

    # Save the trained model
    model_filename = r'C:\Users\mishr\Field-Project-1st-year\trained models\cnn_en.pkl'
    joblib.dump(ensemble_classifier, model_filename)

    # Evaluate the ensemble classifier on validation set
    val_accuracy = ensemble_classifier.score(X_val_features_flat_1, y_val)
    print(f"Ensemble Classifier Validation Accuracy: {val_accuracy}")

    # Plot accuracy of the ensemble classifier
    plt.plot([val_accuracy], marker='o', linestyle='', label='Ensemble Classifier Accuracy')
    plt.title('Ensemble Classifier Validation Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
