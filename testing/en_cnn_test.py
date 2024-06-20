import os
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Constants
IMAGE_SIZE = (224, 224)

# Function to preprocess the input image
def preprocess_image(img_path, target_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (target_size[0], target_size[1]))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

if __name__ == "__main__":
    # Load the trained concatenated model
    model_path = r"C:\Users\mishr\Field-Project-1st-year\trained models\en_cnn.h5"
    model = load_model(model_path)

    # Provide the path to the flower image you want to classify
    flower_img_path = r"C:\Users\mishr\Desktop\sun.jpeg"
    
    # Preprocess the input image and extract features
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    vgg_features = vgg_model.predict(preprocess_image(flower_img_path, IMAGE_SIZE))

    # Dummy Random Forest predictions
    rf_predictions = np.zeros((1, 15))  # Match the shape to (None, 15)
    
    # Use the loaded model to predict the class of the flower image
    predicted_probs = model.predict(np.concatenate((vgg_features.reshape(vgg_features.shape[0], -1), rf_predictions), axis=1))
    predicted_class_index = np.argmax(predicted_probs)
    
    # Get the name of the folder corresponding to the predicted class
    data_dir = r'C:\Users\mishr\Field-Project-1st-year\preprocessed imagedata'
    classes = sorted(os.listdir(data_dir))
    predicted_flower_name = classes[predicted_class_index]

    # Print the predicted flower name
    print("Predicted Flower Name:", predicted_flower_name)
