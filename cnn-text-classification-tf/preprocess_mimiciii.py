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

    def load_from_text(self, data_dir, labelfile_path, stpwd_path, crop_threshold, wiki_dir, shuffle=True):
        """
        input
        -----
        crop_threshold: 
            (portion threshold, length threshold), specify how to crop doc. portion threshold is used whenever it's not
            None
        """
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
        X, Y, admX, doc_lens = [], [], [], []
        for i, (filepath, label_vec, adm_vec) in enumerate(file_labels):
            assert os.path.exists(filepath), 'file not found %s' % filepath

            nc = NoteContainer(filepath, mode=1)
            text_one = nc.fields_asText()

            # stpwd removal
            tokens = learn.preprocessing.tokenizer([text_one]).next()
            filtered_tokens = [tk for tk in tokens if sum([reg(tk) for reg in self.tk_regs]) == 0]
            filtered_tokens = [tk for tk in filtered_tokens if tk not in stpwd]
            doc_lens.append(len(filtered_tokens))

            X.append(filtered_tokens)

            Y.append(label_vec)
            admX.append(adm_vec)

        X, max_doc_len = crop_doc(X, doc_lens, zip(*crop_threshold))
        X = [' '.join(e) for e in X]

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len, min_frequency=3)

        # get all wiki docs
        punc_reg = re.compile(r'[^a-z0-9 ]', flags=re.IGNORECASE)
        space_reg = re.compile(r'\s+')
        wikiX = []
        for i in xrange(len(Y[0])):
            with open(joinpath(wiki_dir, str(i)+'.txt')) as f:
                text_one = f.read()

            text_one = punc_reg.sub(' ', text_one)
            text_one = space_reg.sub(' ', text_one)
            text_one = text_one.strip().lower()

            tokens = self.vocab_processor._tokenizer([text_one]).next()
            filtered_tokens = [tk for tk in tokens if sum([reg(tk) for reg in self.tk_regs]) == 0]
            filtered_tokens = [tk for tk in filtered_tokens if tk not in stpwd]
            wikiX.append(' '.join(filtered_tokens[:max_doc_len]))

        self.vocab_processor.fit(X)
        self.vocab_processor.fit(wikiX)
        self.X = np.array(list(self.vocab_processor.transform(X)))
        self.wikiX = np.array(list(self.vocab_processor.transform(wikiX)))

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


def crop_doc(X, doc_lens, portion_threshold=None, length_threshold=None):

    length_flag = length_threshold is not None
    if length_flag:
        val = length_threshold
    else:
        assert portion_threshold is not None, 'give at least one of the thresholds'
        val = portion_threshold

    hist, bin_edges = np.histogram(doc_lens, bins=100)
    max_doc_len, final_protion = 0, 0
    for i, edge in enumerate(bin_edges):
        cumu_portion = np.sum(hist[:i], dtype=np.float32) / np.sum(hist)
        print('i {}, edge {}, portion {}'.format(i, edge, cumu_portion))

        comp = edge if length_flag else cumu_portion

        if comp > val:
            max_doc_len = int(np.ceil(edge))
            final_protion = cumu_portion
            break

    print '=====>> Using doc_len %i covering %.3f of the totoal docs' % (max_doc_len, final_protion)

    for i in xrange(len(X)):
        X[i] = X[i][:max_doc_len]

    return X, max_doc_len


if __name__ == '__main__':
    loader = DataLoader(0.9, 24)
    loader.load_from_text('../uni_containers_tmp', '../label_index', '../stpwd')
    loader.save_mat('../mat_data')