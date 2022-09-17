import glob

import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
# import ResNetTrain
import numpy as np
import string
import model
from data_util import handle_imagePath

# label = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
from tensorflow.python.keras.layers import GlobalAveragePooling2D

label = list(string.ascii_uppercase + string.ascii_lowercase)

model_save_path = '../model_data/checkpoint/Inception10.ckpt'

model = model.Inception10(num_blocks=2, num_classes=52)
model.load_weights(model_save_path)


# 预测
img_files = glob.glob('../imgs/*.png')
for img_file in img_files:
    img = handle_imagePath(img_file)
    result = model.predict(img)
    print(img_file.split('\\')[-1].split('.')[0], '---', label[np.argmax(result[0])])
