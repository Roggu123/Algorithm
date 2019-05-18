#! /usr/bin/env python
# 进行评价
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters          参数设置
# ==================================================

# Data loading params          数据加载参数
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters        模型超参数设置（隐藏单元的参数）
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters          训练参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# 每一百轮便保存模型
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# 仅保存最近5次模型
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters              设备参数
#当指定的设备不存在时，自动分配（默认为TRUE）
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
#打印日志
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess():
    # Data Preparation
    # 数据准备
    # ==================================================

    # Load data  加载数据，返回数据集(x_text)和标签(y)
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary  生成单词字典
    # 得到最大邮件长度（单词个数），不足的用0填充,依次获取x-text的样本x,对样本x以空格为分隔符，
    # 统计其中的单词个数，将最大的单词个数赋予 max_document_length
    max_document_length = max([len(x.split(" ")) for x in x_text]) # 详见function.md
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length) # 统一长度
    x = np.array(list(vocab_processor.fit_transform(x_text))) # 将句子用id组成的向量来表示，详见Function.md

    # Randomly shuffle data 打乱数据集
    np.random.seed(10) # 详见function.md
    shuffle_indices = np.random.permutation(np.arange(len(y))) # 详见function.md
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set  拆分训练集和测试集
    # TODO: This is very crude, should use cross-validation
    # 负数：从后往前取
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    #依据训练集百分比，找到分界点索引，采取交叉验证会更加合适
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # 定义训练
    # ==================================================

    with tf.Graph().as_default(): # 图Granph()设为默认，详见function.md
        session_conf = tf.ConfigProto(
                                      allow_soft_placement=FLAGS.allow_soft_placement,  # 设置允许TensorFlow在指定设备不存在时自动调整设备,详见funciton.md
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():           # 运行TensorFLow操作（operations）的类,也可以使用上下文管理：with
            cnn = TextCNN(                                # 实例化TextCNN模型，建立模型，另见function.md
                # 句子包含的单词个数，统一句子长度
                sequence_length=x_train.shape[1],
                # 分类个数，导入标签数
                num_classes=y_train.shape[1],
                # 选取进行训练的总单词个数
                vocab_size=len(vocab_processor.vocabulary_),
                # 嵌入层输出
                embedding_size=FLAGS.embedding_dim,
                # 过滤器的行数对应卷积后所得矩阵的列数，由于卷积不同所得矩阵的列数也不同，所以各过滤器行数也不同
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                # 过滤器个数
                num_filters=FLAGS.num_filters,
                # l2惩罚项大小
                l2_reg_lambda=FLAGS.l2_reg_lambda)filter_sizes

            # Define Training procedure 定义训练过程，编译模型
            global_step = tf.Variable(0, name="global_step", trainable=False)       # 统计训练次数
            optimizer = tf.train.AdamOptimizer(1e-3)                                # 指定优化器为Adam
            grads_and_vars = optimizer.compute_gradients(cnn.loss)       # 损失函数偏差与方差即被优化目标
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # train_op是新建的操作，用来对参数做梯度更新，每一次运行train_op就是一次训练

            # Keep track of gradient values and sparsity (optional)
            # 追踪并可视化训练和评估过程
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries  汇总输出储存目录
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy  对损失和准确度的汇总
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries 训练的汇总
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries  评估的汇总
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            # 检查点：               存储模型参数
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            # 检查点存储路径 猜想：out_dir/checkpoints
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary      存储词典
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables  初始化所有参数
            sess.run(tf.global_variables_initializer())

            # 定义单独训练步
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                # 数据将通过占位符节点送给神经网络，必须让所有节点都有值，否则TensorFlow又要报错
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # 通过session.run()运行train_op，返回值就是我们想我评估的操作结果。注意train_op本身没有返回值，它只是更新了网络参数。
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
            
            # 用与单独训练步相似的函数来评估任意数据集的损失和精度，比如验证集或整个训练集。本质上这个函数和之前的一样，但是没有训练操作，也禁用了dropout。
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
