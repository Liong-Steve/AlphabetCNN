# -*-coding:utf-8-*-
import tensorflow as tf

tensorflow_version = tf.__version__
gpu_available = tf.test.is_gpu_available()
print(tf.config.list_physical_devices('GPU'))
print(tensorflow_version)
print(gpu_available)

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 2.0], name="b")
result = tf.add(a, b, name="add")
print(result)