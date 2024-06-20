import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical

# Define the directory containing the data
data_dir = r'C:\Users\mishr\Field-Project-1st-year\preprocessed imagedata'

# Define the desired image size
IMAGE_SIZE = (224, 224)

# Function to load data from a directory
def load_data(directory, image_size):
    X = []
    y = []
    classes = sorted(os.listdir(directory))
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        images = os.listdir(class_dir)
        for image_name in images:
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size[0], image_size[1]))  # Resize images
            X.append(image)
            y.append(class_id)
    return np.array(X), np.array(y)

# Load images and labels
X, y = load_data(data_dir, IMAGE_SIZE)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load VGG16 model without top layers for feature extraction
vgg16_feature_extractor = VGG16(input_shape=(*IMAGE_SIZE, 3), weights='imagenet', include_top=False)

# Freeze VGG16 layers
for layer in vgg16_feature_extractor.layers:
    layer.trainable = False

# Extract features from VGG16 for training and validation sets
vgg16_features_train = vgg16_feature_extractor.predict(X_train)
vgg16_features_val = vgg16_feature_extractor.predict(X_val)

# Train Random Forest classifier on the VGG16 features
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_classifier.fit(vgg16_features_train.reshape(vgg16_features_train.shape[0], -1), y_train)

# Get features from Random Forest predictions
rf_predictions_train = rf_classifier.predict_proba(vgg16_features_train.reshape(vgg16_features_train.shape[0], -1))
rf_predictions_val = rf_classifier.predict_proba(vgg16_features_val.reshape(vgg16_features_val.shape[0], -1))

# Concatenate VGG16 features and RF predictions
concatenated_features_train = np.concatenate((vgg16_features_train.reshape(vgg16_features_train.shape[0], -1), rf_predictions_train), axis=1)
concatenated_features_val = np.concatenate((vgg16_features_val.reshape(vgg16_features_val.shape[0], -1), rf_predictions_val), axis=1)

# Define new VGG16 model with concatenated features
input_features = Input(shape=concatenated_features_train.shape[1:])  # Corrected input shape
x = Dense(1000, activation='sigmoid')(input_features)  # Additional dense layer with sigmoid activation
output = Dense(len(np.unique(y)), activation='softmax')(x)
model = Model(inputs=input_features, outputs=output)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train)
y_val_one_hot = to_categorical(y_val)

# Train the model
history = model.fit(concatenated_features_train, y_train_one_hot,
                    validation_data=(concatenated_features_val, y_val_one_hot),
                    epochs=1000, batch_size=32)

# Plot training and validation accuracy side by side
plt.figure(figsize=(12, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Display final validation accuracy and loss
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])
print("Final Validation Loss:", history.history['val_loss'][-1])

# Save the trained model
model.save(r"C:\Users\mishr\Field-Project-1st-year\trained models\en_cnn.h5")
