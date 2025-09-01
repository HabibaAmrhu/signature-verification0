import os
import random
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Set dataset directory
dataset_dir = "dataset"  # replace with your local dataset path

# Function to collect all images
def collect_images(base_dir):
    exts = ['*.png', '*.jpg', '*.jpeg']
    images = []
    for ext in exts:
        images.extend(glob(os.path.join(base_dir, '**', ext), recursive=True))
    return images

all_imgs = collect_images(dataset_dir)

# Map writers to their images
writers_dict = {}
for img_path in all_imgs:
    filename = os.path.basename(img_path)
    writer_id = filename.split("_")[0]
    writers_dict.setdefault(writer_id, []).append(img_path)

writers = list(writers_dict.keys())

# Function to load and preprocess images
IMG_SIZE = (100, 100)
def load_img(path):
    img = Image.open(path).convert('L').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=-1)

# Generator for image pairs
def pair_generator(batch_size=32):
    while True:
        X1, X2, y = [], [], []
        for _ in range(batch_size):
            if random.random() < 0.5:
                w = random.choice(writers)
                if len(writers_dict[w]) < 2:
                    continue
                imgs = random.sample(writers_dict[w], 2)
                label = 1
            else:
                if len(writers) < 2:
                    continue
                w1, w2 = random.sample(writers, 2)
                imgs = [random.choice(writers_dict[w1]), random.choice(writers_dict[w2])]
                label = 0
