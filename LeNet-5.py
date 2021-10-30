import tensorflow as tf
from mnist_read import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import os


# 导入数据
train_datas, trian_labels = load_mnist('./data/fashion/', 'train')
test_datas, test_labels = load_mnist('./data/fashion/', 't10k')
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 数据处理
train_datas = np.expand_dims(train_datas, axis=3)
test_datas = np.expand_dims(test_datas, axis=3)
train_datas = train_datas/255.0
test_datas = test_datas/255.0
train_datas = tf.image.random_flip_left_right(train_datas)
test_datas = tf.image.random_flip_left_right(test_datas)
# 创建LeNet-5网络
'''
LetNet-5总共7层：输入图像为32x32的灰度图像，Fasion MNSIT为28x28,需进行简单修改（第一层卷积层padding='valid'更换为padding='same'）,具体描述如下：
#layer1:卷积层（卷积核大小：5x5x6, padding='same'，strides=1）, output:28x28x6
#layer2:池化层（卷积核大小：2x2x1, padding='same'， strides=2）, output:14x14x6
#layer3:卷积层（卷积核大小：5x5x16, padding='valid'，strides=1）, output:10x10x16
#layer4:池化层（卷积核大小：2x2x1, padding='same'， strides=2）, output:5x5x16
#layer5:卷积层（卷积核大小：5x5x120, padding='valid'，strides=1）, output:1x1x120
#layer6:全连接层（output:1x84）
#layer7:输出层（output:1x10）
'''
LeNet5_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=6, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Conv2D(filters=192, kernel_size=(4, 4), strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')

    ]
)

# 模型配置
LeNet5_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
# 打印网络参数
print(LeNet5_model.summary())

# tensorboard
# log_dir = os.path.join('logs') #win10下的bug，
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,  histogram_freq=1, batch_size=32,
#     write_graph = False, write_grads=False, write_images=True,
#     embeddings_freq = 0, embeddings_layer_names=None,
#     embeddings_metadata=None, embeddings_data=None, update_freq=500)



#训练
history = LeNet5_model.fit(train_datas, trian_labels, batch_size=64, epochs=10, validation_split=0.1, shuffle=True)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

# 验证测试
res = LeNet5_model.evaluate(test_datas, test_labels)

#  模型保存
LeNet5_model.save('LeNet-5_model.h5')

