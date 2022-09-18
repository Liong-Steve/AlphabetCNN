# coding=utf-8
from matplotlib import pyplot as plt

import string
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image

# 大写字母ascii顺序列表
Alphabet_Upper_Mapping_List = list(string.ascii_uppercase)
Alphabet_Lower_Mapping_List = list(string.ascii_lowercase)
Alphabet_Mapping_List = list(string.ascii_uppercase + string.ascii_lowercase)


# 加载EMNIST数据集
def load_EMNISTdata():
    (train, test), info = tfds.load(
        'emnist',
        split=['train', 'test'],  # 要读取的拆分（例如 'train'、['train', 'test']、'train[80%:]'…）
        data_dir='../tensorflow_datasets',  # 数据集存储的位置（默认为 ~/tensorflow_datasets/）
        shuffle_files=True,  # 控制是否打乱每个周期间的文件顺序（TFDS 以多个较小的文件存储大数据集）
        as_supervised=True,  # 返回包含数据集元数据的 tfds.core.DatasetInfo
        with_info=True,  #
    )
    return (train, test), info


# EMNIST数据集原先是反旋的，需要翻转每张图片数据
def normalize_img(image, label):
    # perm=[1,0,2]表示只翻转第一第二维度
    image = tf.transpose(image, perm=[1, 0, 2])
    return tf.cast(image, tf.float32) / 255.0, label


# 加载翻转并且正则化后的EMNIST数据
def load_normalize_EMNISTdata():
    (train, test), info = load_EMNISTdata()
    # 开始翻转并正则化图片数据
    train = train.map(normalize_img)
    test = test.map(normalize_img)
    return (train, test), info


# 将tensor转为numpy
def tensor_to_numpy(t):
    return np.vstack(tfds.as_numpy(t))


# EMNIST数据集中的label中原先包含数字的label,将所有的label减去10
def normalize_label(image, label):
    return image, label - 10


# 过滤数据集中数字图片
def filter_fn(image, label):
    return tf.math.greater(label, 9)


# 加载EMNIST中的字母数据
def load_normalize_EMNIST_letters():
    (train, test), info = load_normalize_EMNISTdata()
    train = train.filter(filter_fn)
    test = test.filter(filter_fn)
    train = train.map(normalize_label)
    test = test.map(normalize_label)
    return (train, test), info


# 混合大小写标签函数
def combine_label(image, label):
    return image, label % 26


# 加载大小写混合的EMNIST字母数据
def load_normalize_combined_letters():
    (train, test), info = load_normalize_EMNIST_letters()
    train = train.map(combine_label)
    test = test.map(combine_label)
    return (train, test), info


# 将[28,28]的array数据转为[1,28,28,1]的tensorflow数据
def ar_to_tf(img_arr):
    img_tf = tf.reshape(img_arr, [28, 28])
    img_tf = tf.expand_dims(img_tf, axis=-1)
    img_tf = tf.expand_dims(img_tf, axis=0)
    return img_tf


# 二级化图像
def ar_to_arBin(img, threshold=200):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > threshold:
                img[i, j] = 0
            else:
                img[i, j] = 1
    return img


# 处理图片数据,根据图片路径,转换图片后返回tf格式
def handle_imagePath(image_path):
    # 打开图片路径
    img = Image.open(image_path)
    # 显示图片灰度图
    # image = plt.imread(image_path)
    # plt.set_cmap('gray')
    # plt.imshow(image)
    # plt.show()
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L')).astype(np.float32)

    img_arr = ar_to_arBin(img_arr)
    # 显示图片灰度图
    # image = plt.imread(image_path)
    # plt.set_cmap('gray')
    # plt.imshow(image)
    # plt.show()
    img_tf = ar_to_tf(img_arr)
    return img_tf
