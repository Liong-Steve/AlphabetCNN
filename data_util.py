# coding=utf-8
from matplotlib import pyplot as plt

import getConfig
import os
import csv
import string
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image

# 获取 config.txt 文件中的参数配置
# gConfig = {}
# gConfig = getConfig.get_config()

# 获取配置源数据路径
# data_path = gConfig['resource_csv_data']
# if not os.path.exists(data_path):
#     exit()

# 获取配置数据集路径
# x_train_path = gConfig['x_train_path']
# y_train_path = gConfig['y_train_path']

# 大写字母ascii顺序列表
Alphabet_Upper_Mapping_List = list(string.ascii_uppercase)
Alphabet_Lower_Mapping_List = list(string.ascii_lowercase)
Alphabet_Mapping_List = list(string.ascii_uppercase + string.ascii_lowercase)


# 处理源数据
# def handle_csv():
#     print('-----------------Generate Datasets-----------------')
#     # 建立空列表
#     x, y_ = [], []
#     # 读取csv
#     with open(data_path, newline='') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#         # 按行遍历读取内容
#         last_digit_Name = None
#         for row in reader:
#             # 行中第一个数字为大写字母顺序列表
#             digit_Name = row.pop(0)
#             image_array = np.asarray(row)
#             # 将图片数据转为矩阵
#             image_array = image_array.reshape(28, 28)
#             x.append(image_array)  # 图片数据
#             y_.append(digit_Name)  # 标签数据
#             # 打印处理过程信息
#             if last_digit_Name != str(Alphabet_Mapping_List[int(digit_Name)]):
#                 last_digit_Name = str(Alphabet_Mapping_List[int(digit_Name)])
#                 count = 0
#                 print("")
#                 print("Processing Alphabet - " + str(last_digit_Name))
#             count = count + 1
#             if count % 1000 == 0:
#                 print("Images processed: " + str(count))
#     # 变为np.array格式
#     x = np.array(x)
#     y_ = np.array(y_)
#
#     print('-----------------Save Datasets-----------------')
#     # 将矩阵转为行后保存
#     x_save = np.reshape(x, (len(x), -1))
#     np.save(x_train_path, x_save)
#     np.save(y_train_path, y_)
#
#     return x, y_


# 加载处理后的数据
# def load_CSVdata():
#     # 判断是否存在处理好的数据，是则加载，否则处理源数据
#     if os.path.exists(x_train_path) and os.path.exists(y_train_path):
#         print('-----------------Load Datasets-----------------')
#         x_train = np.load(x_train_path)
#         y_train = np.load(y_train_path)
#         # 将图片数据转为矩阵
#         x_train = np.reshape(x_train, (len(x_train), 28, 28))
#     else:
#         x_train, y_train = handle_csv()
#     # 返回处理后的数据
#     return x_train, y_train


# 加载EMNIST数据集
def load_EMNISTdata():
    (train, test), info = tfds.load(
        'emnist',
        split=['train', 'test'],
        data_dir='./tensorflow_datasets',
        shuffle_files=True,  #
        as_supervised=True,
        with_info=True,
    )
    return (train, test), info


# EMNIST数据集原先是反旋的，需要翻转每张图片数据
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.transpose(image, perm=[1, 0, 2])
    return tf.cast(image, tf.float32) / 255.0, label
    # # perm=[1,0,2]表示只翻转第一第二维度
    # return tf.transpose(image, perm=[1, 0, 2]), label


# 加载翻转并且正则化后的EMNIST数据
def load_normalize_EMNISTdata():
    (train, test), info = load_EMNISTdata()
    # 开始翻转并正则化图片数据
    # train = train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # test = test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
    #
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L')).astype(np.float32)

    img_arr = ar_to_arBin(img_arr)
    # 显示图片灰度图
    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.show()
    # for i in range(28):
    #     for j in range(28):
    #         if img_arr[i][j] < 200:
    #             img_arr[i][j] = 255
    #         else:
    #             img_arr[i][j] = 0
    #
    # img_arr = img_arr / 255.
    img_tf = ar_to_tf(img_arr)
    return img_tf
