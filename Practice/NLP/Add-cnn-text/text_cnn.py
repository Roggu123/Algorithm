import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      #初始化参数，包括序列长度（句子包含单词个数），类别数，输入总单词个数，嵌入层大小（输出），过滤器种类数，每一类过滤器包含的个数，正则化项
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout 设置输入，输出，正则单元的占位符
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)    l2范数
        l2_loss = tf.constant(0.0)

        # Embedding layer 嵌入层，与图像CNN不同之处
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W存储全部word vector的矩阵，W初始化时是随机random出来的
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # 查找input_x中所有的ids，获取它们的word vector，得到的embedded_chars的shape应该是[None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 输入的word vectors得到后，输入到卷积层（函数：tf.nn.conv2d），由于输入参数还差一个差一个in_channels，使用expand dim来适应conv2d的input要求，embedded_chars后面加一个in_channels=1
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        # 针对不同滤波器矩阵行数filter size（过滤的单词个数）产生对应的卷积和池化，不同滤波器提取不同特征
        pooled_outputs = []
        # filter_size表示滤波器过滤的单词个数
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer 卷积层
                # 滤波器的形状，过滤单词个数*单词维度*滤波器个数，过滤器个数等于一批训练的容量，即一批数据包含多少句子
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W为滤波器矩阵
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # b可理解为阈值
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # 定义卷积
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity 应用非线性变换
                # 隐层采用了relu激活函数
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # 定义池化
                pooled = tf.nn.max_pool(
                    h,
                    # 池化矩阵长度根据滤波器扫描单词个数而变,即不同种类滤波器对应不同池化层
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # 将每一种滤波器对应的池化结果（1*num_filters）添加在pooled_outputs后面
                # pooled_outputs最终结果为 （1*num_filters*len(filter_size)）
                pooled_outputs.append(pooled)

        # Combine all the pooled features 整合所有池化后的特征，一维矩阵，元素个数为滤波器种类数
        # 总滤波器个数为一种滤波器个数*滤波器类数(num_filters*len(filter_sizes))
        num_filters_total = num_filters * len(filter_sizes)
        # pooled_outputs是列表，包含有len(filter_sizes)个pooled，3表示按列拼接，即组合成二维矩阵[1*number_filters*len(filter_sizes)]，第一行表示第一个句子在不同滤波器下显示的特征
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 将h_pool转换为num_filters_total列的矩阵,实际上就是将特征向量平展，每一批训练都平展为一维向量
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        # 添加辍学层，是最流行的卷积神经网络正则化方法，辍学层随机“禁用”其神经元的一部分。这可以防止神经元共同适应并迫使它们学习单独有用的特征。
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        # 使用max-pooling中的特征向量（应用了dropout），我们可以通过矩阵乘法和选择具有最高分数的类来生成预测。我们还可以应用softmax函数将原始分数转换为标准化概率，但这不会改变我们的最终预测。
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        # 损失和准确性
        # 使用我们的分数，我们可以定义损失函数。损失是我们网络错误的衡量标准，我们的目标是最小化它。分类的标准损失函数问题是交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
