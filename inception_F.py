import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mnist_read import load_mnist
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
# train_datas = tf.image.random_flip_left_right(train_datas)
# test_datas = tf.image.random_flip_left_right(test_datas)

image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,  # 布尔值，使输入数据集去中心化（均值为0）, 按feature执行。
    featurewise_std_normalization=True,  #  布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行。
    # rotation_range=45,  # 随机旋转角度范围。随机45度旋转
    width_shift_range=.15,  # 随机宽度偏移量
    height_shift_range=.15,  # 随机高度偏移
    horizontal_flip=True,  # 是否随机水平翻转
    # zoom_range=0.3  # 调整缩放范围。将图像随机缩放阈量30％
)
image_gen_train.fit(train_datas)


class Inception(tf.keras.layers.Layer):
    def __init__(self,c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = tf.keras.layers.Conv2D(c1, kernel_size=1, activation='relu', padding='same')
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], kernel_size=1, padding='same', activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], kernel_size=3, padding='same',
                              activation='relu')
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], kernel_size=1, padding='same', activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], kernel_size=5, padding='same',
                              activation='relu')
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same', strides=1)
        self.p4_2 = tf.keras.layers.Conv2D(c4, kernel_size=1, padding='same', activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return tf.concat([p1, p2, p3, p4], axis=-1)  # 在通道维上连结输出


def creat_model():
    net1 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=24, kernel_size=(5, 1), strides=1, padding='valid', use_bias=False),
        tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 5), strides=1, padding='valid', use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Inception(16, (6, 16), (6, 16), 16),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 1), strides=1, padding='valid', use_bias=False),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 5), strides=1, padding='valid', use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    net1.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    return net1

model = creat_model()
print(model.summary())

# 创建保存文件
check_path = 'checkpoint/model.ckpt'
check_dir = os.path.dirname(check_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(check_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)

# tensorboard
# log_dir = os.path.join('logs') # win10下的bug，
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,  histogram_freq=1, batch_size=32,
#     write_graph = False, write_grads=False, write_images=True,
#     embeddings_freq = 0, embeddings_layer_names=None,
#     embeddings_metadata=None, embeddings_data=None, update_freq=500)

#训练监视器
# monitor_callback = tf.keras.callbacks.EarlyStopping( monitor='val_loss',  verbose=1, mode='auto',
#     baseline=None, restore_best_weights=False)

net = creat_model()
# 载入模型，重新训练时注释掉即可
# latest = tf.train.latest_checkpoint(check_dir)
# net.load_weights(latest)

# 训练
history = net.fit(train_datas, trian_labels, batch_size=256, epochs=20,
                  validation_split=0.1, shuffle=True)
                  # callbacks=[tensorboard_callback])

# 画图
marker_on=[5,10,15]
plt.plot(history.history['accuracy'], color='black',label='training_acc')
plt.plot(history.history['val_accuracy'], label='validation_acc',color='black', linestyle = '-', marker='v',markevery=marker_on, markersize=5)
plt.plot(history.history['loss'], label='train_loss', color='black',linestyle = '-', marker='x',markevery=marker_on, markersize=5)
plt.plot(history.history['val_loss'], label='val_loss', color='black', linestyle = '-', marker='s',markevery=marker_on, markersize=5)
plt.legend(loc='center right')
plt.title('ILIC Model', fontname='Times New Roman', fontsize=20, weight='bold', style='italic')
plt.xlabel('epoch', fontname='Times New Roman', fontsize=15)
plt.ylabel('accuracy/loss', fontname='Times New Roman', fontsize=15)
x_major_locator=MultipleLocator(2)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(-0.5,20)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(-0.1,1)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
plt.show()


# 验证测试
res = net.evaluate(test_datas, test_labels)







