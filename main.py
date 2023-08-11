from flask import Flask, render_template, request, send_file, redirect
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.train import Checkpoint, latest_checkpoint
from model.cae import CAE
from utils import load_image, parse_np_array_image

app = Flask(__name__)

# Load the CAE model and the latest checkpoint
model = CAE()
checkpoint_path = 'bener/'  # Replace with the actual path to your model checkpoint
ckpt = Checkpoint(transformer=model)
ckpt.restore(latest_checkpoint(checkpoint_path)).expect_partial()

def get_image_size(image_path):
    return os.path.getsize(image_path)

def convert_bytes(size, unit=None):
    # Fungsi ini akan mengkonversi ukuran dari byte menjadi KB, MB, atau GB
    # unit bisa berisi "KB", "MB", atau "GB" (default adalah None)
    if unit == "KB":
        return size / 1024
    elif unit == "MB":
        return size / (1024 ** 2)
    elif unit == "GB":
        return size / (1024 ** 3)
    else:
        return size

# Function to compress the image and save it to the server
def compress_and_save_image(image, image_path):
    compressed_image = compress(model, image)
    decompressed_image = decompress(model, compressed_image)
    decompressed_image_array = decompressed_image[0].numpy()
    decompressed_image_array = (decompressed_image_array * 255.0).astype(np.uint8)
    decompressed_image_pil = Image.fromarray(decompressed_image_array)
    decompressed_image_pil.save(image_path)

# Function to compress the image
def compress(model, image):
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor

# Function to decompress the image
def decompress(model, compressed_image):
    return model(compressed_image)

# Additional function to handle image decompression and return the decompressed image path
def decompress_image(image_path):
    # Load the compressed image from the specified path
    compressed_image = load_image(image_path)
    # Compress the image using the model and then decompress it
    compressed_tensor = compress(model, compressed_image)
    decompressed_image = decompress(model, compressed_tensor)

    # Save the decompressed image to a temporary location
    decompressed_image_path = 'static/decompressed_image.jpg'
    decompressed_image_array = decompressed_image[0].numpy()
    decompressed_image_array = (decompressed_image_array * 255.0).astype(np.uint8)
    decompressed_image_pil = Image.fromarray(decompressed_image_array)
    decompressed_image_pil.save(decompressed_image_path)

    return decompressed_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return "No image file found."

        image = request.files['image']
        if image.filename == '':
            return "No selected image."

        # Save the uploaded image to a temporary location
        temp_image_path = 'static/temp_image.jpg'
        image.save(temp_image_path)

        # Load the image from the temporary location
        image = load_image(temp_image_path)

        # Process the image (compression and decompression)
        compress_and_save_image(image, 'static/compressed_image.jpg')

        # Get image sizes
        original_image_size = get_image_size(temp_image_path)
        compressed_image_size = get_image_size('static/compressed_image.jpg')

        # Convert sizes to KB
        original_image_size = convert_bytes(original_image_size, unit="KB")
        compressed_image_size = convert_bytes(compressed_image_size, unit="KB")

        # Paths for displaying images on the page
        original_image = 'static/temp_image.jpg'
        compressed_image = 'static/compressed_image.jpg'

        return render_template('index.html', original_image=original_image, compressed_image=compressed_image,
                               original_image_size=original_image_size, compressed_image_size=compressed_image_size)

    return render_template('index.html', original_image=None, compressed_image=None)

@app.route('/decompress', methods=['POST'])
def decompress_route():
    if 'compressed_image' not in request.files:
        return "No compressed image file found."

    compressed_image = request.files['compressed_image']
    if compressed_image.filename == '':
        return "No selected compressed image."

    # Save the uploaded compressed image to a temporary location
    temp_compressed_image_path = 'static/temp_compressed_image.jpg'
    compressed_image.save(temp_compressed_image_path)

    # Perform image decompression and get the path of the decompressed image
    decompressed_image_path = decompress_image(temp_compressed_image_path)

    # Get image sizes
    original_image_size = get_image_size('static/temp_image.jpg')
    compressed_image_size = get_image_size(temp_compressed_image_path)
    decompressed_image_size = get_image_size(decompressed_image_path)

    # Convert sizes to KB
    original_image_size = convert_bytes(original_image_size, unit="KB")
    compressed_image_size = convert_bytes(compressed_image_size, unit="KB")
    decompressed_image_size = convert_bytes(decompressed_image_size, unit="KB")

    # Paths for displaying images on the page
    original_image = 'static/temp_image.jpg'
    compressed_image = 'static/compressed_image.jpg'
    decompressed_image = decompressed_image_path

    return render_template('index.html', original_image=original_image, compressed_image=compressed_image,
                           decompressed_image=decompressed_image, original_image_size=original_image_size,
                           compressed_image_size=compressed_image_size, decompressed_image_size=decompressed_image_size)

@app.route('/back-to-home', methods=['GET'])
def back_to_home():
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
