import cv2
import numpy as np
import os
from PIL import Image

# Define the paths to the two folders containing the grayscale images
folder1_path = '../cropped_dataset/inputs'
folder2_path = '../cropped_dataset/semantic_annotations'

# Get a list of the file names in folder1 and folder2
folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

output_folder = '../cropped_dataset/concat_inputs'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Iterate over the file names and concatenate the images with the same file name
for file_name in folder1_files:
    if file_name in folder2_files:
        # Load the grayscale images from folder1 and folder2
        img1_path = os.path.join(folder1_path, file_name)
        img2_path = os.path.join(folder2_path, file_name)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        img_concat = np.concatenate(
            (img1[..., np.newaxis], img2[..., np.newaxis]), axis=2)
        img = Image.fromarray(img_concat)
        # Save the concatenated image
        output_path = os.path.join(output_folder, file_name)
        img.save(output_path)
