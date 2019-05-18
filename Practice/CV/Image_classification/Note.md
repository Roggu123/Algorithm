# 构建模型，先配置模型的层，再编译模型
model = keras.Sequential([
<<<<<<< HEAD
<br>&nbsp;&nbsp;&nbsp;keras.layers.Flatten(input_shape=(28, 28)),  # 将图像格式从二维数组（28x28 像素）转换成一维数组（28 * 28 = 784 像素），转为灰度图
<br>&nbsp;&nbsp;&nbsp;keras.layers.Dense(128, activation=tf.nn.relu),  # 全连接层，使用了ReLU激活函数<br>&nbsp;&nbsp;&nbsp;keras.layers.Dense(10, activation=tf.nn.softmax)    # 全连接层，最后一层，用于分类
<br>])

