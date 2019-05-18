import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # 将非字母，非数字，非标点符号的字符替换为空格
    string = re.sub(r"\'s", " \'s", string)                # 替换“'s”为“ ‘s”,如“It's”被替换为“It 's”
    string = re.sub(r"\'ve", " \'ve", string)              #替换“‘ve”为“ ‘ve”,如“I've”被替换为“I 've”
    string = re.sub(r"n\'t", " n\'t", string)              #替换“n't”为“ n't”,如“don't”被替换为“do n't”
    string = re.sub(r"\'re", " \'re", string)           #替换“‘re”为“ 're”，如“you're”被替换为“you 're”
    string = re.sub(r"\'d", " \'d", string)                #替换“'d”为“ 'd”，如“I'd”被替换为“I 'd”
    string = re.sub(r"\'ll", " \'ll", string)              #替换“'ll”为“ 'll”，如“I'll”被替换为“I 'll”
    string = re.sub(r",", " , ", string)#替换“,”为“ ,”，如“It is good,but”被替换为“It is good , but”
    string = re.sub(r"!", " ! ", string)                #替换“!”为“ ! ”，如“Great!”被替换为“Great ! ”
    string = re.sub(r"\(", " \( ", string)              #替换“(”为“ ( ”，如“2(to)”被替换为“2 （ to ) ”
    string = re.sub(r"\)", " \) ", string)              #同上
    string = re.sub(r"\?", " \? ", string)#替换“?”为“ ？”，如“Really?”被替换为“Really ? ”
    string = re.sub(r"\s{2,}", " ", string)                 #将两个及以上的空白字符替换为一个空格
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
