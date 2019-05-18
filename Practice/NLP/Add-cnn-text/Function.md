# <center>第3章 Function(*函数*)</center>
**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**  
Explain how the python files play a role in this project.（*针对第2章中Python文件作用的具体解释*）

## 3.1 data_helpers.py
1. **clean\_str(string)**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;替换非字母，非数字，非标点符号的字符为空格；替换缩写字符；替换标点符号等。  
<font color=FF0000>除了正则表达式可用于字符匹配，大小写转换外，还有什么办法可进行上述操作？</font>   
[关于python 的re.sub用法](https://blog.csdn.net/lovemianmian/article/details/8867613)  
[正则表达式中各种字符的含义](https://www.cnblogs.com/afarmer/archive/2011/08/29/2158860.html)  
[常用正则表达式](https://www.jianshu.com/p/0cb001fe3572)  
2. **load\_data\_and\_labels(positive\_data\_file, negative\_data\_file)**  
3. **batch\_iter(data, batch\_size, num\_epochs, shuffle=True)**

## 3.2 Train.py
### 3.2.1 preprocess()
+ <font color=#0000ff>**fit_transform()**</font>  
`x = np.array(list(vocab_processor.fit_transform(x_text)))`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>搭建了一个最基础的CNN模型</font>**，有输入层（*input layer*），卷积层（*convolutional layer*），最大池化层（*max-pooling layer*）和最后输出的softmax layer。  
参考：[文本预处理方法小记](https://zhuanlan.zhihu.com/p/31767633) 
 
+  <font color=#0000ff>**len()**</font>  
`max_document_length = max([len(x.split(" ")) for x in x_text])`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;提供有关**<font color=#FF0000>数据的处理的方法</font>**，加载数据（*load\_data\_and\_labels()*）、清理数据（*clean\_str()*），定义批迭代器（每次训练时使用不同的批次）（*batch\_iter()*）
   
+ <font color=#0000ff>**seed()**</font>  
`np.random.seed(10)`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>搭建了一个最基础的CNN模型</font>**，有输入层（*input layer*），卷积层（*convolutional layer*），最大池化层（*max-pooling layer*）和最后输出的softmax layer。  
参考：[Python seed() 函数](http://www.runoob.com/python/func-number-seed.html)

+ <font color=#0000ff>**permutation()**</font>  
`shuffle_indices = np.random.permutation(np.arange(len(y)))`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>训练模型及真正处理数据</font>**，其中包含数据准备（*preprocess()*）即数据加载，字典生成和数据预处理、  
参考：[Numpy.random中shuffle与permutation的区别](https://blog.csdn.net/lyy14011305/article/details/76207327)  

### 3.2.2 train() 
+ <font color=#0000ff>**Graph()**</font>  
`with tf.Graph().as_default():`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>一个图(Graph)包含操作和张量</font>**，每个程序中可以含有多个图，但大多数程序也只需要一个图就够了。我们可以在多个session中重复使用一个图，但是不能在一个session中调用多个图。  
参考：[TensorFlow学习（三）：Graph和Session](https://blog.csdn.net/xierhacker/article/details/53860379)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)

+ <font color=#0000ff>**allow\_soft\_placement**</font>  
`allow_soft_placement=FLAGS.allow_soft_placement,`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>设置允许TensorFlow在指定设备不存在时自动调整设备</font>**。例如，如果我们的代码把一个操作放在GPU上，但又在一台没有GPU的机器上运行，如果没有allow_soft_placement就会报错。  
参考：[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)
  
+ <font color=#0000ff>**Session()**</font>
  
		sess = tf.Session(config=session_conf)  
		with sess.as_default():  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>运行TensorFLow操作（operations）的类 </font>**，一个Seesion包含了操作对象执行的环境。是执行计算图操作所在的环境，包含变量和队列的状态。每个Session执行一个图。  
参考：[TensorFlow学习（三）：Graph和Session](https://blog.csdn.net/xierhacker/article/details/53860379)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)
   
+ <font color=#0000ff>**cnn = TextCNN（）**</font>
  
		cnn = TextCNN(
        	sequence_length=x_train.shape[1],
          		num_classes=y_train.shape[1],
         	     vocab_size=len(vocab_processor.vocabulary_),
          	 embedding_size=FLAGS.embedding_dim,
               filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
              l2_reg_lambda=FLAGS.l2_reg_lambda)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>实例化TextCNN模型 </font>**，所有定义的变量和操作就会被放进默认的计算图和会话。  
参考：[TensorFlow学习（三）：Graph和Session](https://blog.csdn.net/xierhacker/article/details/53860379)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)----TextCNN类

+ <font color=#0000ff>**grad_summaries**</font>

		grad_summaries = []
		for g, v in grads_and_vars:
    		if g is not None:
        		grad_hist_summary =	tf.summary.histogram("{}/grad/hist".format(v.name), g)
        		sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), 
        			tf.nn.zero_fraction(g))
        		grad_summaries.append(grad_hist_summary)
        		grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>追踪并可视化训练和评估过程</font>**，这里我们分别追踪训练和评估的汇总，有些量是重复的，但又很多量是只在训练过程中想看的（比如参数更新值）。tf.merge_summary函数可以很方便地把合并多个汇总融合到一个操作。  
参考：[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)
	
### 3.2.3 train_step（）  
```python
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
      _, step, summaries, loss, accuracy = sess.run(
          [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
          feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      train_summary_writer.add_summary(summaries, step)
```  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>定义一个单独的训练步</font>**，来评估模型在一批数据上的表现，并相应地更新参数。打印出本轮训练的损失函数和精确度，存储汇总至磁盘。
  
参考：[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)

+ <font color=#0000ff>**feed_dict**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>输入数据到神经网络</font>**，feed_dict里的数据将通过占位符节点送给神经网络，必须让所有节点都有值，否则TensorFlow将会报错。  
+ <font color=#0000ff>**session.run()**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>通过session.run()运行train_op</font>**，返回值就是我们想我评估的操作结果。注意train_op本身没有返回值，它只是更新了网络参数。最后我们打印出本轮训练的损失函数和精确度，存储汇总至磁盘。  
  
### 3.2.4 dev_step（）
```python
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
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>定义一个单独的评估步</font>**，用与单独训练步相似的函数来评估任意数据集的损失和精度，比如验证集或整个训练集。本质上这个函数和之前的一样，但是没有训练操作，也禁用了dropout。
  
参考：[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)

### 3.2.5 batches
```python
batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
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
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>定义一个训练循环</font>**，一批一批的在数据上迭代，调用train_step函数，然后评估并存储检查点。
  
参考：[基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)

+ <font color=#0000ff>**batch_iter**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>给数据进行分批</font>**。
+ <font color=#0000ff>**tf.train.global_step**</font>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<font color=#FF0000>返回global_step的值</font>**。依据global_step的值判断是否进行评估和储存。

## 3.3 eval.py
### 3.3.1 数据准备  
+ <font color=#0000ff>;**加载数据：**</font>
	
		x_raw, y_test =data_helpers.load_data_and_labels(FLAGS.positive_data_file, 
			FLAGS.negative_data_file)
    	y_test = np.argmax(y_test, axis=1)
    
+ <font color=#0000ff>;**将数据映射到字典：**</font> 

		vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
		vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
		x_test = np.array(list(vocab_processor.transform(x_raw)))
		
### 3.3.2 进行评价
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;加载图与会话，获取相关数据，定义评价循环，获取预测值，获取模型精确度（评价），保存预测数据。
  
## 3.4 text_cnn.py
### 3.4.1 
### 3.4.2 浪里个浪 
参考：[用Tensorflow实现CNN文本分类(详细解释及TextCNN代码解释)](http://www.voidcn.com/article/p-sjhkchtl-bmr.html)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)原始论文
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)原始论文导读
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)原始代码解析
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/) 理解卷积神经网络 
- [用Tensorflow实现CNN文本分类(详细解释及TextCNN代码解释)](http://www.voidcn.com/article/p-sjhkchtl-bmr.html)对代码中textCNN的理解 3.4 
- [CNN 实现文本分类](https://github.com/fengxqinx/TextCNN)  主要讲解卷积神经网络在文本分类中的实现(代码其参考)
- [文本分类(下)-卷积神经网络(CNN)在文本分类上的应用](https://juejin.im/post/5b584ae1e51d4517580dfd56)主要讲解卷积神经网络对文本分类原理及实现  
- [基于TensorFlow用卷积神经网络做文本分类](https://juejin.im/post/5a72909bf265da3e377c5a32)主要介绍文本分类的训练进程 3.2
