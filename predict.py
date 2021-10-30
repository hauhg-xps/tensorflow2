from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 导入模型/加入inception模块后无法保存为.h5文件，无法进行预测。
new_model = keras.models.load_model('LeNet-5_model.h5')


# 从网上找12张图片,获取预测样本
plt.figure(figsize=(12, 8))
for i in range(12):
    img = Image.open("picter/{}.png".format(str(i)))
    # img = img.convert("L")
    # img = np.array(img)
    shape = img.size
    img = img.resize((28, 28))

    plt.subplot(3, 4, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    plt.title("Original size:{},{}".format(str(shape[0]), str(shape[1])))
    plt.xlabel(i)
plt.tight_layout()
plt.show()

# 检测12张图片并可视化结果

# 载入数据
def loadImage(i):
    from PIL import Image, ImageOps

    img = Image.open("picter/{}.png".format(str(i)))
    img = img.convert("L")  # 将彩色图像转换成黑白图像
    img = ImageOps.invert(img)  # 将图像黑色和白色反转，即为了与Fashion MNIST数据集保持一致，将背景部分转换成白色
    img = img.resize((28, 28))  # 改变大小为28*28
    img = np.array(img)  # 转换成NumPy数据形式
    # img = img / 255.0

    return img


def plot_image(i, img, prediction):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction)

    plt.title(i + 1)
    plt.xlabel("Predict:{} {:2.0f}% ".format(classNames[predicted_label], 100 * np.max(prediction)))


def plot_value_array(i, img, prediction):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction[0], color="#777777")
    plt.ylim([0, 1])
    # predicted_label = np.argmax(prediction[0])


num_rows = 4
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))


for i in range(num_images):

    img = loadImage(i)
    # print(img.shape)
    img = img.reshape((-1, 28, 28, 1))
    prediction = new_model.predict(img)

    img = img.reshape((28, 28))
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)  # 绘制原始图像
    plot_image(i, img, prediction)

    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)  # 绘制预测概率分布图
    plot_value_array(i, img, prediction)
plt.tight_layout()
plt.show()