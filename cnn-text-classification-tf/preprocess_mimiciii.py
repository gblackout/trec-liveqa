import numpy as np
import os
from tensorflow.contrib import learn
from os.path import join as joinpath
from process.generate_data import NoteContainer
import random
from utils.misc import makedir
import re


class DataLoader:
    def __init__(self, data_partition, batch_size):
        """
        
        input
        -----
        data_partition:
            float between (0,1) indicating the ratio of training data in whole dataset
            
        """
        self.X = None
        self.Y = None
        self.data_partition = data_partition
        self.partition_ind = None
        self.vocab_processor = None
        self.med_freq = None
        self.batch_size = batch_size

        self.max_doc_len = None
        self.vocab_size = None
        self.num_class = None

        # reg for cleaning filtering out meaningless tokens
        reg1 = re.compile(r'\d')           # remove tokens that are just/contains digits
        reg2 = re.compile(r'^\w{1,3}$')     # remove tokens that are too short (len<=3)
        self.tk_regs = [lambda x:reg1.search(x) is not None,
                        lambda x:reg2.match(x) is not None]

    def load_from_text(self, data_dir, labelfile_path, stpwd_path, max_doc_len=None, threshold=0.7, shuffle=True):
        assert os.path.isfile(labelfile_path), 'index file not found at %s' % labelfile_path
        assert os.path.isdir(data_dir), 'data dir not found at %s' % data_dir
        assert os.path.isfile(stpwd_path), 'stopwords file not found at %s' % stpwd_path

        # get all filenames and their labels
        file_labels = []
        with open(labelfile_path) as f:
            for line in f:
                parts = line.split()

                filepath = parts[0]
                mask_vec = parts[1:]

                file_labels.append([joinpath(data_dir, filepath), map(int, mask_vec)])

        # get set of stpwd
        stpwd = set()
        with open(stpwd_path) as f:
            stpwd.update([line.strip() for line in f])

        if max_doc_len is None:
            assert threshold is not None, 'max_doc_len and threshold cannot be both None'
            max_doc_len, thre_hist = get_doc_len(file_labels, threshold=threshold)
            print '=====>> Using doc_len as %i with accumulated histogram %.3f' % (max_doc_len, thre_hist)

        if shuffle:
            random.shuffle(file_labels)

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len, min_frequency=3)

        # visit all json files, extract X and Y
        X, Y = [], []
        for i, (filepath, label_vec) in enumerate(file_labels):
            assert os.path.exists(filepath), 'file not found %s' % filepath

            nc = NoteContainer(filepath, mode=1)
            text_one = nc.fields_asText()[:max_doc_len]

            # stpwd removal
            tokens = self.vocab_processor._tokenizer([text_one]).next()
            filtered_tokens = [tk for tk in tokens if sum([reg(tk) for reg in self.tk_regs]) == 0]
            filtered_tokens = [tk for tk in filtered_tokens if tk not in stpwd]

            X.append(' '.join(filtered_tokens))
            Y.append(label_vec)

        self.X = np.array(list(self.vocab_processor.fit_transform(X)))
        self.Y = np.array(Y, dtype=np.float32)

        self.vocab_size = len(self.vocab_processor.vocabulary_)
        self.max_doc_len = max_doc_len
        self.partition_ind = int(self.X.shape[0] * self.data_partition)
        self.num_class = self.Y.shape[1]

        assert self.vocab_size > 100
        print("Vocabulary Size       : {:d}".format(self.vocab_size))

    def load_from_mat(self, mat_dir):
        assert os.path.isdir(mat_dir)
        self.X = np.load(joinpath(mat_dir, 'X.npy'))
        self.Y = np.load(joinpath(mat_dir, 'Y.npy'))
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(joinpath(mat_dir, 'vocab'))

        self.vocab_size = len(self.vocab_processor.vocabulary_)
        self.max_doc_len = self.X.shape[1]
        self.partition_ind = int(self.X.shape[0] * self.data_partition)
        self.num_class = self.Y.shape[1]

    def batcher(self, train=True, batch_size=None):
        bsize = batch_size if batch_size else self.batch_size
        if train:
            bias = 0
            data_size = self.partition_ind
            num_batches = DataLoader.compute_numOf_batch(data_size, bsize)
        else:
            bias = self.partition_ind
            data_size = self.X.shape[0] - self.partition_ind
            num_batches = DataLoader.compute_numOf_batch(data_size, bsize)

        while True:
            for batch_num in range(num_batches):
                start_index = bias + batch_num * bsize
                end_index = bias + min((batch_num + 1) * bsize, bias + data_size)
                is_epochComplete = batch_num + 1 == num_batches

                x_batch, y_batch = self.get_chunk(start_index, end_index)

                yield x_batch, y_batch, batch_num, is_epochComplete

    def get_chunk(self, start_index, end_index):
        return self.X[start_index:end_index], self.Y[start_index:end_index]

    @staticmethod
    def compute_numOf_batch(total_size, batch_size):
        return int((total_size - 1) / batch_size) + 1

    def save_mat(self, mat_dir):
        assert not ((self.X is None) or (self.Y is None) or (self.vocab_processor is None))
        makedir(mat_dir)
        np.save(joinpath(mat_dir, 'X'), self.X)
        np.save(joinpath(mat_dir, 'Y'), self.Y)
        self.vocab_processor.save(joinpath(mat_dir, 'vocab'))

    def get_med_freq(self):
        if self.med_freq is None:
            self.med_freq = np.sum(self.Y[self.partition_ind:, :], axis=0)

        return self.med_freq


def get_doc_len(file_labels, threshold=0.8):
    doc_lens = []
    for filepath, _ in file_labels:
        assert os.path.isfile(filepath), 'file not found at %s' % filepath

        nc = NoteContainer(filepath, mode=1)
        text_one = nc.fields_asText()

        doc_lens.append(len(text_one.split()))

    hist, bin_edges = np.histogram(doc_lens)
    for i, edge in enumerate(bin_edges):
        print('i {}, edge {}, portion {}'.format(i, edge, float(np.sum(hist[:i])) / float(np.sum(hist))))

        if float(np.sum(hist[:i])) / float(np.sum(hist)) > threshold:
            thre_edge = int(edge)
            thre_hist = float(np.sum(hist[:i])) / float(np.sum(hist))
            break

    return thre_edge, thre_hist


def load_data(data_dir, labelfile_path, vocab_path, max_doc_len=None, threshold=0.7, shuffle=True):
    """
    main method for obtaining dataset
    
    input
    -----
    data_dir: 
        directory containing the dataset
    labelfile_path:
        path to index file containing label and filepath
    vocab_path:
        previous VocabularyProcessor save path, if None then one is created at that path
    max_doc_len:
        max num of WORDS allowed in doc, if None then inferred by threshold
    threshold:
        threshold of distribution of doc length with which we infer the max_doc_len
    shuffle:
        set true to shuffle data when loading
    
    output
    -----
    X:
        np.int64, num_samples * max_doc_len 
    Y:
        np.float32, num_samples * num_classes
    max_doc_len:
        max num of WORDS allowed in doc    
    
    """

    assert os.path.isfile(labelfile_path), 'index file not found at %s' % labelfile_path

    file_labels = []
    with open(labelfile_path) as f:
        for line in f:
            parts = line.split()

            filepath = parts[0]
            mask_vec = parts[1:]

            file_labels.append([joinpath(data_dir, filepath), map(int, mask_vec)])

    if max_doc_len is None:
        assert threshold is not None, 'max_doc_len and threshold cannot be both None'
        max_doc_len, thre_hist = get_doc_len(file_labels, threshold=threshold)
        print('=====>> Using doc_len as {} with accumulated histogram {:g}\n'.format(max_doc_len, thre_hist))

    if shuffle:
        random.shuffle(file_labels)

    if vocab_path and os.path.isfile(vocab_path):
        print '=====>> vocab file found'
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        assert len(vocab_processor.vocabulary_) > 100
        save_flag = False
    else:
        print '=====>> vocab file not found, creating new one...'
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len, min_frequency=3)
        save_flag = True

    # visit all json files, extract X and Y
    X, Y = [], []
    for i, (filepath, label_vec) in enumerate(file_labels):
        assert os.path.exists(filepath), 'file not found %s' % filepath

        nc = NoteContainer(filepath, mode=1)
        text_one = nc.fields_asText()[:max_doc_len]

        X.append(text_one)
        Y.append(label_vec)

    if save_flag:
        vocab_processor.fit(X)
        assert len(vocab_processor.vocabulary_) > 100
        vocab_processor.save(vocab_path)

    X = np.array(list(vocab_processor.transform(X)))
    Y = np.array(Y, dtype=np.float32)
    print("Vocabulary Size       : {:d}".format(len(vocab_processor.vocabulary_)))

    return X, Y, max_doc_len


def batch_iter(x, y, batch_size, shuffle=True):
    """
    
    input (shuffled)
    ------
    x: 
        np.array(num_samples, max_doc_len) integer vector 
    y: 
        (num_samples, num_classes)    
    
    return
    -----
    x_batch: 
        np.array(batch_size, max_doc_len)
    y_batch: 
        np.array(batch_size, num_classes)
        
    """
    data_size = len(x)  # array can have len()
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    while True:
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x  # np.array(num_samples, 1014)
            y_shuffled = y  # np.array(num_samples, 2)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            is_epochComplete = batch_num + 1 == num_batches_per_epoch

            x_batch, y_batch = get_batched(x_shuffled, y_shuffled, start_index, end_index)

            yield x_batch, y_batch, batch_num, is_epochComplete


def get_batched(x, y, start_index, end_index):
    """
    # input: x : np.array(num_samples, max_doc_len), y : np.array(num_samples, num_classes), batch_start_index, batch_end_index
    # return: x_batch: np.array(batch_size, max_doc_len), y_batch: np.array(batch_size, num_classes)
    """
    x_batch = x[start_index:end_index]
    y_batch = y[start_index:end_index]
    return x_batch, y_batch


if __name__ == '__main__':
    loader = DataLoader(0.9, 24)
    loader.load_from_text('../uni_containers_tmp', '../label_index', '../stpwd')
    loader.save_mat('../mat_data')