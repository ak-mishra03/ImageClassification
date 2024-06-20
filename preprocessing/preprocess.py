import os
import cv2
import numpy as np
from tqdm import tqdm

# Define constants
IMAGE_SIZE = 200  # You can adjust this according to your requirements
KERNEL_SIZE = (3, 3)  # Size of the Gaussian kernel for blur
FLOWER_TYPES = ['Balloon Flower', 'Barbeton Daisy', 'Bishop of llandaff', 'Black-eyed Susan', 'Canterbury Bells', 'Daffodils', 'Globe Thistle', 'Marigold', 'Mexican Aster', 'Oxeye Daisy', 'Red Ginger', 'Silverbush', 'Sunflower', 'Water lily', 'Windflower']
DATA_DIRECTORY = r"C:\Users\mishr\Desktop\field project\flower data"  # Directory where your images are stored
OUTPUT_DIRECTORY = r"C:\Users\mishr\Desktop\output"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

from PIL import Image
def preprocess_image(image_path):
    # Load image using PIL
    pil_img = Image.open(image_path)
    # Convert image to RGB format (removing alpha channel)
    rgb_img = pil_img.convert('RGB')
    # Convert PIL image to numpy array
    img = np.array(rgb_img)
    
    # Resize image
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Apply Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, KERNEL_SIZE, 0)
    
    # Rescale pixel values to [0, 255]
    img = img * 255.0
    
    return img
for flower_type in FLOWER_TYPES:
    print(f"Preprocessing images for {flower_type}...")
    input_folder = os.path.join(DATA_DIRECTORY, flower_type)
    output_folder = os.path.join(OUTPUT_DIRECTORY, flower_type)

    # Create output directory for the specific flower type
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through images in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter out only image files
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            preprocessed_img = preprocess_image(image_path)
            cv2.imwrite(output_path, preprocessed_img)

print("Image preprocessing completed.")
