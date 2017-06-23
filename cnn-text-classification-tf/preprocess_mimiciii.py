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
        self.admX = None
        self.wikiX = None
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

    def load_from_text(self, data_dir, labelfile_path, stpwd_path, wiki_dir, max_doc_len=None, threshold=0.7, shuffle=True):

        assert os.path.isfile(labelfile_path), 'index file not found at %s' % labelfile_path
        assert os.path.isdir(data_dir), 'data dir not found at %s' % data_dir
        assert os.path.isfile(stpwd_path), 'stopwords file not found at %s' % stpwd_path
        assert os.path.isdir(wiki_dir), 'wiki dir not found at %s' % wiki_dir


        # get all filenames and their labels
        file_labels = []
        with open(labelfile_path) as f:
            for line in f:

                # TODO ad hoc solution
                if ';' in line:
                    parts = line.split(';')
                    adm_vec = map(int, parts[1].strip().split())
                    mask_vec = map(int, parts[2].strip().split())
                else:
                    parts = line.split()
                    mask_vec = map(int, parts[1:])
                    adm_vec = None

                filepath = parts[0].strip()

                file_labels.append([joinpath(data_dir, filepath), mask_vec, adm_vec])

        # get set of stpwd
        stpwd = set()
        with open(stpwd_path) as f:
            stpwd.update([line.strip() for line in f])

        if shuffle:
            random.shuffle(file_labels)

        # visit all json files, extract X and Y
        X, Y, admX = [], [], []
        for i, (filepath, label_vec, adm_vec) in enumerate(file_labels):
            assert os.path.exists(filepath), 'file not found %s' % filepath

            nc = NoteContainer(filepath, mode=1)
            text_one = nc.fields_asText()

            # stpwd removal
            tokens = learn.preprocessing.tokenizer([text_one]).next()
            filtered_tokens = [tk for tk in tokens if sum([reg(tk) for reg in self.tk_regs]) == 0]
            filtered_tokens = [tk for tk in filtered_tokens if tk not in stpwd]

            X.append(' '.join(filtered_tokens))
            Y.append(label_vec)
            admX.append(adm_vec)

        if max_doc_len is None:
            assert threshold is not None, 'max_doc_len and threshold cannot be both None'
            max_doc_len, thre_hist = get_doc_len(X, threshold=threshold)
            print '=====>> Using doc_len as %i with accumulated histogram %.3f' % (max_doc_len, thre_hist)

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len, min_frequency=3)

        # # get all wiki docs
        # punc_reg = re.compile(r'[^a-z0-9 ]', flags=re.IGNORECASE)
        # space_reg = re.compile(r'\s+')
        # wikiX = []
        # for i, filename in enumerate(os.listdir(wiki_dir)):
        #     with open(joinpath(filename, wiki_dir)) as f:
        #         text_one = f.read()[:max_doc_len]
        #
        #     text_one = punc_reg.sub(' ', text_one)
        #     text_one = space_reg.sub(' ', text_one)
        #     text_one = text_one.strip().lower()
        #
        #     tokens = self.vocab_processor._tokenizer([text_one]).next()
        #     filtered_tokens = [tk for tk in tokens if sum([reg(tk) for reg in self.tk_regs]) == 0]
        #     filtered_tokens = [tk for tk in filtered_tokens if tk not in stpwd]
        #     wikiX.append(' '.join(filtered_tokens))

        self.vocab_processor.fit(X)
        # self.vocab_processor.fit(wikiX)
        self.X = np.array(list(self.vocab_processor.transform(X)))
        # self.wikiX = np.array(list(self.vocab_processor.transform(wikiX)))

        self.Y = np.array(Y, dtype=np.float32)
        if admX[0] is not None:
            self.admX = np.array(admX, dtype=np.float32)

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

                x_batch, y_batch, adm_batch = self.get_chunk(start_index, end_index)

                yield x_batch, y_batch, adm_batch, batch_num, is_epochComplete

    def get_chunk(self, start_index, end_index):
        return self.X[start_index:end_index], self.Y[start_index:end_index], \
               self.admX[start_index:end_index] if self.admX is not None else None

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


def get_doc_len(X, threshold=0.8):
    doc_lens = []
    # TODO ac hoc
    for text in X:
        doc_lens.append(len(text.split()))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.hist(doc_lens, bins=100)
    plt.tight_layout()
    plt.savefig('doc_dist')

    import sys
    sys.exit(0)


    # hist, bin_edges = np.histogram(doc_lens)
    # for i, edge in enumerate(bin_edges):
    #     print('i {}, edge {}, portion {}'.format(i, edge, float(np.sum(hist[:i])) / float(np.sum(hist))))
    #
    #     if float(np.sum(hist[:i])) / float(np.sum(hist)) > threshold:
    #         thre_edge = int(edge)
    #         thre_hist = float(np.sum(hist[:i])) / float(np.sum(hist))
    #         break

    return None, None


if __name__ == '__main__':
    loader = DataLoader(0.9, 24)
    loader.load_from_text('../uni_containers_tmp', '../label_index', '../stpwd')
    loader.save_mat('../mat_data')