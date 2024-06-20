import os
import cv2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications import VGG16
import joblib  # For loading the pre-trained model

# Constants
IMAGE_SIZE = (100, 100)

# Function to preprocess the input image
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

if __name__ == "__main__":
    # Load the trained model
    model_path = r"C:\Users\mishr\Field-Project-1st-year\trained models\cnn_ml.pkl"
    model = joblib.load(model_path)

    # Provide the path to the flower image you want to classify
    flower_img_path = r"C:\Users\mishr\Desktop\sun.jpeg"
    
    # Preprocess the input image
    preprocessed_img = preprocess_image(flower_img_path, IMAGE_SIZE)

    # Load the VGG16 model for feature extraction
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Extract features using VGG16 model
    img_features = vgg_model.predict(preprocessed_img)

    # Flatten the extracted features
    flattened_img = img_features.reshape(1, -1)

    # Use the loaded model to predict the class of the flower image
    predicted_class_index = model.predict(flattened_img)
    
    # Get the name of the folder corresponding to the predicted class
    data_dir = r'C:\Users\mishr\Field-Project-1st-year\preprocessed imagedata'
    classes = sorted(os.listdir(data_dir))
    predicted_flower_name = classes[int(predicted_class_index)]

    # Print the predicted flower name
    print("Predicted Flower Name:", predicted_flower_name)
