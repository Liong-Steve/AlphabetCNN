import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import string
from PIL import Image

# 标签数据 对应 字符
# 0-9
# 10-35 A~Z
# 36-61 a~z
label = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
# 记录已经遍历的数量
records = [0 for x in label]
# 存储文件夹
path = 'D:\\Temp\\emnist\\'

# 加载数据
(train, test), ds_info = tfds.load(
    'emnist',
    split=['train', 'test'],
    data_dir='./tensorflow_datasets',
    shuffle_files=True,  #
    as_supervised=True,
    with_info=True,
)


# emnist数据集天生旋转，翻转每张图片数据
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    # image = tf.transpose(image, perm=[1, 0, 2])
    # return tf.cast(image, tf.float32) / 255., label
    # perm=[1,0,2]表示只翻转第一第二维度
    return tf.transpose(image, perm=[1, 0, 2]), label


# 开始翻转图片数据
train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test = test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

# 将图片数据转为numpy
train_numpy = np.vstack(tfds.as_numpy(train))
test_numpy = np.vstack(tfds.as_numpy(test))

# 保存图片
for p in train_numpy:
    image_array = np.asarray(p[0])
    image_array = image_array.reshape(28, 28)
    image = Image.fromarray(image_array.astype('uint8'))
    lab = p[1]
    directory = label[lab]
    if lab >= 36:
        directory += directory
    path_train = path + '\\train\\' + directory + '\\' + label[lab] + '-' + str(records[lab]) + '.png'
    image.save(path_train)
    records[lab] += 1

for p in test_numpy:
    image_array = np.asarray(p[0])
    image_array = image_array.reshape(28, 28)
    image = Image.fromarray(image_array.astype('uint8'))
    lab = p[1]
    directory = label[lab]
    if lab >= 36:
        directory += directory
    path_train = path + '\\test\\' + directory + '\\' + label[lab] + '-' + str(records[lab]) + '.png'
    image.save(path_train)
    records[lab] += 1
