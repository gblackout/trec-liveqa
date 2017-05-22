import numpy as np
import os
from tensorflow.contrib import learn
import sys
sys.path.append('../')
from get_mimiciii_data import get_admission_text_from_json


def get_doc_len(dataset_dir, paths, threshold=0.8):
    doc_lens = []
    for path in paths:
        # assert os.path.exists(dataset_dir+path)
        if not os.path.exists(dataset_dir + path):
            continue
        text_one = get_admission_text_from_json(dataset_dir + path)
        doc_lens.append(len(text_one.split()))
    hist, bin_edges = np.histogram(doc_lens)
    for i, edge in enumerate(bin_edges):
        print('i {}, edge {}, portion {}'.format(i, edge, float(np.sum(hist[:i])) / float(np.sum(hist))))
        if float(np.sum(hist[:i])) / float(np.sum(hist)) > threshold:
            thre_edge = int(edge)
            thre_hist = float(np.sum(hist[:i])) / float(np.sum(hist))
            break
    return thre_edge, thre_hist


def load_mimiciii_adMed(dataset_dir, file_labels_fn, adMed_file_labels_fn, max_doc_len=None, threshold=None):
    # return two lists: examples and labels
    # examples is a list of strings of the same number of words, labels is a list of label one hot list
    filename = dataset_dir + file_labels_fn
    f = open(filename, 'r').read().split('\n')
    file_labels = []
    for line in f:
        if not line: continue
        parts = line.split(',')
        if len(parts) != 2:
            print(len(parts), parts[0])
        assert len(parts) == 2, '[Error] len(parts):{}, parts[0]:{}'.format(len(parts), parts[0])
        file_labels.append([parts[0].strip(), [int(float(i)) for i in parts[1].split()]])

    num_adMed_labels = None
    filename = dataset_dir + adMed_file_labels_fn
    f = open(filename, 'r').read().split('\n')
    file_adMed_labels = {}
    for line in f:
        if not line: continue
        parts = line.split(',')
        if len(parts) != 2:
            print(len(parts), parts[0])
        assert len(parts) == 2, '[Error] len(parts):{}, parts[0]:{}'.format(len(parts), parts[0])
        file_adMed_labels[parts[0].split('/')[-1].strip()] = [int(float(i)) for i in parts[1].split()]
        if num_adMed_labels is None: num_adMed_labels = len(parts[1].split())

    if max_doc_len is None:
        if threshold is None: threshold = 0.7
        print('=====>> Computing doc_len with threshold as {}'.format(threshold))
        max_doc_len, thre_hist = get_doc_len(dataset_dir, [item[0] for item in file_labels], threshold=threshold)
        print('=====>> Using doc_len as {} with accumulated histogram {:g}\n'.format(max_doc_len, thre_hist))

    examples, adMed_labels, labels, count = [], [], [], 0
    for i, (filename, label_vec) in enumerate(file_labels):
        filepath = dataset_dir + filename
        if not os.path.exists(filepath): continue
        text_one = get_admission_text_from_json(filepath)
        text_one = ' '.join(text_one.split()[:max_doc_len])
        examples.append(text_one)
        labels.append(label_vec)
        if filename.split('/')[-1].strip() in file_adMed_labels:
            adMed_labels.append(file_adMed_labels[filename.split('/')[-1].strip()])
            count += 1
        else:
            adMed_labels.append([0] * num_adMed_labels)
    print('[INFO] Among {} samples, {} have admission medications'.format(len(file_labels), count))
    return examples, adMed_labels, labels, max_doc_len

def load_mimiciii(dataset_dir, file_labels_fn, max_doc_len=None, threshold=None):
    # return two lists: examples and labels
    # examples is a list of strings of the same number of words, labels is a list of label one hot list
    filename = dataset_dir + file_labels_fn
    f = open(filename, 'r').read().split('\n')
    file_labels = []
    for line in f:
        if not line: continue
        parts = line.split(',')
        if len(parts) != 2:
            print(len(parts), parts[0])
        assert len(parts) == 2, '[Error] len(parts):{}, parts[0]:{}'.format(len(parts), parts[0])
        ### For old version of file_label file, it uses files in notes_json.
        ### For new version, change it to use files in notes_json_all_fields
        if 'notes_json_all_fields' not in parts[0].strip():
            modified_filename = parts[0].strip().replace('notes_json', 'notes_json_all_fields')
        file_labels.append([modified_filename, [int(float(i)) for i in parts[1].split()]])

    if max_doc_len is None:
        if threshold is None: threshold = 0.7
        print('=====>> Computing doc_len with threshold as {}'.format(threshold))
        max_doc_len, thre_hist = get_doc_len(dataset_dir, [item[0] for item in file_labels], threshold=threshold)
        print('=====>> Using doc_len as {} with accumulated histogram {:g}\n'.format(max_doc_len, thre_hist))

    examples, labels = [], []
    for i, (filepath, label_vec) in enumerate(file_labels):
        filepath = dataset_dir + filepath
        if not os.path.exists(filepath): continue
        text_one = get_admission_text_from_json(filepath)
        text_one = ' '.join(text_one.split()[:max_doc_len])
        examples.append(text_one)
        labels.append(label_vec)
    return examples, labels, max_doc_len

def load_data(dataset_dir, file_labels_fn, max_doc_len=None, load_adMed=False, adMed_file_labels_fn=None, threshold=None):
    if not load_adMed:
        examples, labels, max_doc_len = load_mimiciii(dataset_dir=dataset_dir, file_labels_fn=file_labels_fn, max_doc_len=max_doc_len, threshold=threshold)
        x = examples
        y = np.array(labels, dtype=np.float32)
        return x, y, max_doc_len
    else:
        assert adMed_file_labels_fn, '[Error] Must provide adMed_file_labels_fn file name if set load_adMed as true'
        examples, adMed_labels, labels, max_doc_len = load_mimiciii_adMed(dataset_dir=dataset_dir, file_labels_fn=file_labels_fn,
                                        adMed_file_labels_fn=adMed_file_labels_fn, max_doc_len=max_doc_len, threshold=threshold)
        x = examples
        adMed_labels = np.array(adMed_labels, np.float32)
        y = np.array(labels, dtype=np.float32)    # y shape: (num_samples, num_classes)
        return x, adMed_labels, y, max_doc_len   # return [ np.array(num_samples, max_char_len), np.array(num_samples, num_classes)]


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    input (shuffled): x: np.array(num_samples, max_doc_len) integer vector, y: (num_samples, num_classes)
    return: x_batch: np.array(batch_size, max_doc_len), y_batch: np.array(batch_size, num_classes)
    """
    data_size = len(x)      # array can have len()
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x      # np.array(num_samples, 1014)
            y_shuffled = y      # np.array(num_samples, 2)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched(x_shuffled, y_shuffled, start_index, end_index)
            yield x_batch, y_batch  # [np.array(batch_size, max_doc_len), np.array(batch_size, num_classes))]


def get_batched(x, y, start_index, end_index):
    """
    # input: x : np.array(num_samples, max_doc_len), y : np.array(num_samples, num_classes), batch_start_index, batch_end_index
    # return: x_batch: np.array(batch_size, max_doc_len), y_batch: np.array(batch_size, num_classes)
    """
    x_batch = x[start_index:end_index]
    y_batch = y[start_index:end_index]
    return x_batch, y_batch

def batch_iter_adMed(x, y, admed, batch_size, num_epochs, shuffle=True):
    """
    input (shuffled): x: np.array(num_samples, max_doc_len) integer vector, y: (num_samples, num_classes)
    return: x_batch: np.array(batch_size, max_doc_len), y_batch: np.array(batch_size, num_classes)
    """
    data_size = len(x)      # array can have len()
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            admed_shuffled = admed[shuffle_indices]
        else:
            x_shuffled = x      # np.array(num_samples, 1014)
            y_shuffled = y      # np.array(num_samples, 2)
            admed_shuffled = admed
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch, admed_batch = get_batched_adMed(x_shuffled, y_shuffled, admed_shuffled, start_index, end_index)
            yield x_batch, y_batch, admed_batch  # [np.array(batch_size,max_doc_len), np.array(batch_size, num_classes))]


def get_batched_adMed(x, y, admed, start_index, end_index):
    """
    # input: x : np.array(num_samples, max_doc_len), y : np.array(num_samples, num_classes), batch_start_index, batch_end_index
    # return: x_batch: np.array(batch_size, max_doc_len), y_batch: np.array(batch_size, num_classes)
    """
    x_batch = x[start_index:end_index]
    y_batch = y[start_index:end_index]
    admed_batch = admed[start_index:end_index]
    return x_batch, y_batch, admed_batch