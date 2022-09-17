import glob

import numpy as np
import string
import model
from data_util import handle_imagePath

# label = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
label = list(string.ascii_uppercase + string.ascii_lowercase)

model_save_path = '../model_data/checkpoint/ResNet18.ckpt'

model = model.ResNet18([2, 2, 2, 2], num_classes=52)
model.load_weights(model_save_path)


# 预测
img_files = glob.glob('../imgs/*.png')
for img_file in img_files:
    img = handle_imagePath(img_file)
    result = model.predict(img)
    print(img_file.split('\\')[-1].split('.')[0], '---', label[np.argmax(result[0])])
