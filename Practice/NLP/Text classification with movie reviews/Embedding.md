# <center><font color=#7CFC00 face="黑体">第1章 Embedding 层</font></center>
```python
keras.layers.embeddings.Embedding(input_dim,
	output_dim,embeddings_initializer='uniform',
	embeddings_regularizer=None,
	activity_regularizer=None,
	embeddings_constraint=None,
	mask_zero=False,input_length=None)
```
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**嵌入层**将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]。
## 1.1 参数解释
+ **input_dim**：大或等于0的整数，字典长度，即输入数据最大下标+1,即单词个数，一般会取前1000个或500个等，不一定全部取； 
+ **output_dim**：大于0的整数，代表全连接嵌入的维度，即用该维度表示一个单词；
+ **embeddings_initializer**: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers；  
+ **embeddings_regularizer**: 嵌入矩阵的正则项，为Regularizer对象 ;
+ **embeddings_constraint**: 嵌入矩阵的约束项，为Constraints对象 ;
+ **mask_zero**：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为$|vocabulary| + 2$;  
+ **input\_length**：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**对应代码解析:**

```Python
    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
```

该嵌入层仅使用了前两个参数，设置输入数据（input\_dim）为1000个，每个单词用16维向量表示（output\_dim）。
## 1.2 方法流程解析：
### 1.2.1 单词转词向量流程：
![Alt text](https://img-blog.csdn.net/20170824172635016?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamlhbmdwZW5nNTk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast "转词向量") 
(1)提取文章所有的单词，把其按其出现的次数降序(取前50000个)，如单词‘network’出现的次数最多，编号ID为0，依次类推$\dots$
  
(2)每个编号ID都可以使用50000维的二进制(one-hot);

(3)最后，我们会生产一个矩阵M，行大小为词的个数50000，列大小为词向量的维度(通常取128或300)，比如矩阵的第一行就是编号ID=0，即network对应的词向量。  
*上述过程实现了整数数组转张量中第二种方法，即填充数组（pad_sequence）方法。
### 1.2.2 模型（方法）解析：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用模型word2Vec**获取矩阵$M$**,通过学习文本来用词向量的形式表征词的语义信息。而Embedding嵌入层其实也就是一种映射，嵌入一个新的空间，将原来的单词映射到一个新的多维空间。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Word2Vec模型中，主要有Skip-Gram和CBOW两种模型，从直观上理解，Skip-Gram是给定input word来预测上下文。而CBOW是给定上下文，来预测input word。本例子中使用的是Skip-Gram模型。
![Alt text](https://static.leiphone.com/uploads/new/article/740_740/201706/594b306c8b3b1.png?imageMogr2/format/jpg/quality/90 "Word2Vec模型") 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;模型分为两个部分，**第一部分为建立模型，第二部分是通过模型获取嵌入词向量**。建模过程为基于训练数据构建一个神经网络，该过程称为“Fake Task”，获取嵌入词向量过程为模型通过训练数据学得参数。该方法也会在无监督特征学习中使用。

（1） **建模**
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;若有一个句子“The dog barked at the mailman”：

* 首先我们选一个词作为输入词，如选取“dog”作为input word；  
* 有了input word以后，再定义一个参数skip\_window，它代表着从当前input word的一侧（左边或右边）选取词的数量。若设置skip\_window=2，从'dog'右侧选取词，那么最终获得窗口中的词（包括input word在内）就是['The', 'dog'，'barked', 'at']。skip\_window=2代表着选取左input word左侧2个词和右侧2个词进入窗口，所以整个窗口大小span=2x2=4。另一个参数叫num\_skips，它代表着从窗口中选取多少个不同的词作为output word，当skip\_window=2，num\_skips=2时，将会得到两组 (input word, output word) 形式的训练数据，即 ('dog', 'barked')，('dog', 'the')；当num_skips=3时，则将会得到三组（input word,output word）形式的训练数据，即（‘dog’,'barked'），（‘dog’,'the'），（‘dog’，‘at’）。
* 神经网络基于这些训练数据将会输出一个概率分布，这个概率代表着词典中的每个词是output word的可能性。例如，第二步设置skip_window和num_skips=2的情况下获得了两组训练数据。先拿一组数据 ('dog', 'barked') 来训练神经网络，那么模型通过学习这个训练样本，会告诉我们词汇表中每个单词是“barked”的概率大小。
![Alt text](https://static.leiphone.com/uploads/new/article/740_740/201706/594b319eb5f1f.png?imageMogr2/format/jpg/quality/90 "例子1") 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;模型从每对单词出现的次数中习得统计结果,神经网络会得到更多常见单词组的样本对，而非常见单词组的样本对很少见到。因此当模型完成训练后，给定一个单词”Soviet“作为输入，输出的结果中”Union“或者”Russia“要比”Sasquatch“被赋予更高的概率。  

（2） **模型细节** 

* **输入层**
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;神经网络只能接受数值输入，不可能把一个单词字符串作为输入，因此要基于训练文档来构建自己的词汇表（vocabulary）再对单词进行one-hot编码。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;举例，继续使用“The dog barked at the mailman”，基于这个句子，可以构建一个大小为5的词汇表（忽略大小写和标点符号）：("the", "dog", "barked", "at", "mailman")，我们对这个词汇表的单词进行编号0-4。那么”dog“就可以被表示为一个5维向量[0, 1, 0, 0, 0]。
![Alt text](https://static.leiphone.com/uploads/new/article/740_740/201706/594b31d0920ef.png?imageMogr2/format/jpg/quality/90 "例子2") 
* **隐层**
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入层输入的是10000维单词向量，我们想将用300个特征来表示单词（即用300维向量表示单词），设置隐层的权重矩阵为10000行，300列（隐层有300个结点）。根据矩阵乘法，权重矩阵可与输入单词向量矩阵运算，最终得到的权重矩阵每一行表示一个单词向量。
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;看下面的图片，左右两张图分别从不同角度代表了输入层-隐层的权重矩阵。左图中每一列代表一个10000维的词向量和隐层单个神经元连接的权重向量。从右边的图来看，每一行实际上代表了每个单词的词向量。**我们最终的目标就是学习这个隐层的权重矩阵**。
![Alt text](https://static.leiphone.com/uploads/new/article/740_740/201706/594b320f8ed60.png?imageMogr2/format/jpg/quality/90 "例子3") 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入被one-hot编码以后大多数维度上都是0（实际上仅有一个位置为1），如果进行矩阵相乘会消耗大量资源，因此进行矩阵计算时，直接去查输入向量中取值为1的维度下对应的那些权重值（行）。隐层的输出就是每个输入单词的“嵌入词向量”。**其实到这一步，Embedding层的任务已经完成**。
![Alt text](https://static.leiphone.com/uploads/new/article/740_740/201706/594b322ae0c72.png?imageMogr2/format/jpg/quality/90 "例子4") 
* **输出层**
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;经过神经网络隐层的计算，词向量从一个1 x 10000的向量变成1 x 300的向量，再被输入到输出层。输出层是一个softmax回归分类器，它的每个结点将会输出一个0-1之间的值（概率），这些所有输出层神经元结点的概率之和为1。
![Alt text](https://static.leiphone.com/uploads/new/article/740_740/201706/594b3267c64f4.png?imageMogr2/format/jpg/quality/90 "例子5")

## 1.3 参考文献：
AI研习社：[”一文详解 Word2vec 之 Skip-Gram 模型（结构篇）“] (https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html "With a Title")， https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html （2019/2/10）


