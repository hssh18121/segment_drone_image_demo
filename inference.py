import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from numpy import array, expand_dims, argmax, zeros, uint8
from PIL import Image
from tensorflow.keras.models import load_model
from patchify import patchify, unpatchify
import argparse
import time

print("Starting executable file...")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Segmentation model inference')
parser.add_argument('original_image', type=str, help='Path to the original image')
parser.add_argument('model_path', type=str, help='Path to the trained model')
parser.add_argument('output_folder', type=str, help='Output folder for the prediction image')

args = parser.parse_args()

# Load the image
img = imread(args.original_image, 1)
img = cvtColor(img, COLOR_BGR2RGB)

# Load the model
model = load_model(args.model_path, compile=False)

# Size of patches
patch_size = 256
# Number of classes 
n_classes = 5

large_img = Image.fromarray(img)
large_img = array(large_img)     

# Start the timer
start_time = time.time()

# Predict patch by patch with no smooth blending
SIZE_X = (img.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
SIZE_Y = (img.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size

large_img = Image.fromarray(img)
large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
large_img = array(large_img)
print(large_img.shape)

patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap
patches_img = patches_img[:, :, 0, :, :, :]
print(patches_img.shape)

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, :, :, :]
        
        # Use MinMaxScaler instead of just dividing by 255.
        # single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = expand_dims(single_patch_img, axis=0)
        pred = model.predict(single_patch_img)
        pred = argmax(pred, axis=3)
        pred = pred[0, :, :]
        
        patched_prediction.append(pred)

patched_prediction = array(patched_prediction)
print(patched_prediction.shape)  # Should be (number_of_patches, patch_height, patch_width)

# Reshape to 4D array
patched_prediction = patched_prediction.reshape((patches_img.shape[0], patches_img.shape[1], patch_size, patch_size))

# Unpatchify to get the large image
unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

# End the timer
end_time = time.time()

# Calculate and print the total time taken
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")

# Define a color map for the pixel values
color_map = {
    0: [0, 0, 0],         # Black for background
    1: [0, 255, 0],       # Green for tree
    2: [0, 0, 255],       # Blue for water
    3: [255, 255, 0],     # Yellow for road
    4: [255, 0, 0]        # Red for building
}

# Get the shape of the unpatched_prediction
height, width = unpatched_prediction.shape

# Create an empty array for the colored image
colored_image = zeros((height, width, 3), dtype=uint8)

# Apply the color map
for value, color in color_map.items():
    colored_image[unpatched_prediction == value] = color

# Convert the numpy array to a PIL Image
image = Image.fromarray(colored_image)

# Derive prediction file name
base_name = os.path.basename(args.original_image)
name, ext = os.path.splitext(base_name)
prediction_file = f'{name}_segmented.tif'

# Specify the folder and filename
filepath = os.path.join(args.output_folder, prediction_file)

# Ensure the directory exists
os.makedirs(args.output_folder, exist_ok=True)

# Save the image
image.save(filepath)

print(f'Image saved to {filepath}')
