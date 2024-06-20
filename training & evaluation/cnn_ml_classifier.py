import os
import numpy as np
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2
import matplotlib.pyplot as plt
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

    # Load VGG16 model pre-trained on ImageNet without the top classification layer
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Extract features using VGG16 model
    X_train_features = vgg_model.predict(X_train)
    X_val_features = vgg_model.predict(X_val)

    # Reshape features
    X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_val_features_flat = X_val_features.reshape(X_val_features.shape[0], -1)

    # Define and train SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_features_flat, y_train)

    # Save the trained model
    model_filename = r'C:\Users\mishr\Field-Project-1st-year\trained models\cnn_ml.pkl'
    joblib.dump(svm_classifier, model_filename)

    # Evaluate SVM classifier on validation set
    svm_accuracy = svm_classifier.score(X_val_features_flat, y_val)
    print(f"SVM Validation Accuracy: {svm_accuracy}")

    # Plot accuracy and loss for the final result
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.axhline(y=svm_accuracy, color='r', linestyle='--', label='SVM Accuracy')
    plt.title('Final Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss (no loss information available for SVM classifier)
    plt.subplot(1, 2, 2)
    plt.title('Final Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()
