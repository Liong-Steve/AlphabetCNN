import tensorflow as tf
import os
from matplotlib import pyplot as plt

import data_util
import model

# np.set_printoptions(threshold=np.inf)

SHUFFLE_SIZE = 500
BATCH_SIZE = 128
# 加载正则化的EMNIST训练数据和测试数据
(train, test), info = data_util.load_normalize_EMNIST_letters()

# 缓存数据
train = train.cache()
test = test.cache()
# 打乱数据，并组合成批次
train = train.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
test = test.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
# 无限重复数据
train = train.repeat()
test = test.repeat()
# 数据预取，减小延迟提高吞吐率
train = train.prefetch(tf.data.experimental.AUTOTUNE)
test = test.prefetch(tf.data.experimental.AUTOTUNE)


model = model.AlexNet8(num_classes=52)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'],
              )

checkpoint_save_path = "../model_data/checkpoint/AlexNet8.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(train,
                    epochs=50,
                    validation_data=test,
                    validation_steps=1000,
                    steps_per_epoch=500,
                    validation_freq=1,
                    callbacks=[cp_callback])
model.summary()


###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
