#! /usr/bin/env python
import math
import os
import sys
import time

from utils.misc import linesep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf, numpy as np
from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score, jaccard_similarity_score, hamming_loss

sys.path.append('../')
import preprocess_mimiciii
# from text_cnn_deep import TextCNNDeep
from text_cnn import TextCNN
# from text_char_cnn import CharCNN

# Parameters
# ==================================================

# Load files
tf.flags.DEFINE_string("file_labels_fn", '../../label_index', "")

tf.flags.DEFINE_string("vocab_load_file", '../medication_output/vocab', "vocabulary file")
tf.flags.DEFINE_integer("build_vocabulary", 1, "load vocabulary file")
tf.flags.DEFINE_integer("load_model", 0, "load model file")
tf.flags.DEFINE_string("load_model_folder", './runs_sigmoid_1512/1494012307/checkpoints', "load model file")
tf.flags.DEFINE_integer("load_train_data", 0, "load saved training data")
tf.flags.DEFINE_string("load_traindata_file", '../medication_output/traindata', "load saved training data")

# Folders
tf.flags.DEFINE_string("task", "medication", "")
tf.flags.DEFINE_integer("multilabel", 1, "softmax or sigmoid")

tf.flags.DEFINE_string("dataset_dir", '../../uni_containers_tmp', "")

tf.flags.DEFINE_integer("max_doc_len", None,
                        "The maximum number of words in a document, each word is transformed to an integer in vector representation.")
tf.flags.DEFINE_string("log_dir", 'logs/', "")
tf.flags.DEFINE_string("features_dir", 'features', "")
tf.flags.DEFINE_integer("save_features", 0, "")
tf.flags.DEFINE_integer("save_features_per_epoch", 5, "")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("learning_rate_decay", 0.9, "learning rate decay")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 24, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_per_epoch", 1, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_per_batch", 0, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_per_epoch", 5, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
FLAGS.load_traindata_file = FLAGS.load_traindata_file + '_full.pkl' if '_small_' not in FLAGS.file_labels_fn else FLAGS.load_traindata_file + '_small.pkl'
traindata_savepath = '../medication_output/traindata_full.pkl' if '_small_' not in FLAGS.file_labels_fn else '../medication_output/traindata_small.pkl'

print("\nParameters...")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def logitics(x):
    return math.exp(-np.logaddexp(0, -x))


def get_prediction_sigmoid(scores, threshold=0.5):
    predictions = np.zeros_like(scores)
    for row, logits_v in enumerate(scores):
        for col, value in enumerate(logits_v):
            if logitics(value) >= threshold:
                predictions[row][col] = 1
            else:
                predictions[row][col] = 0
    return predictions


def get_prediction_softmax(scores):
    return np.argmax(scores, axis=1)


def compute_confusion_matrix(confusion_matrix):
    temp1 = sum(confusion_matrix[c][c] for c in range(num_classes))
    temp2 = sum(confusion_matrix[i][j] for i in range(num_classes) for j in range(num_classes))
    accuracy = float(temp1) / float(temp2)
    precision, recall, F1 = [], [], []
    for c in range(num_classes):
        if sum(confusion_matrix[c][i] for i in range(num_classes)) == 0:
            precision.append(0.0)
        else:
            precision.append(
                float(confusion_matrix[c][c]) / float(sum(confusion_matrix[c][i] for i in range(num_classes))))

        if sum(confusion_matrix[i][c] for i in range(num_classes)) == 0:
            recall.append(0.0)
        else:
            recall.append(
                float(confusion_matrix[c][c]) / float(sum(confusion_matrix[i][c] for i in range(num_classes))))

        F1.append(
            0 if (precision[c] + recall[c]) == 0 else 2.0 * (precision[c] * recall[c]) / (precision[c] + recall[c]))
    overall_precision = sum(precision) / float(num_classes)
    overall_recall = sum(recall) / float(num_classes)
    if overall_precision + overall_recall == 0:
        overall_F1 = 0.0
    else:
        overall_F1 = 2.0 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    return accuracy, precision, recall, F1, overall_precision, overall_recall, overall_F1


def get_classification_accuracy(predictions, labels):
    # input predictions and labels are 1-D array
    assert predictions.shape == labels.shape
    accuracy = float(sum(1 for p, l in zip(predictions, labels) if p == l)) / float(predictions.shape[0])
    confusion_matrix = np.zeros((num_classes, num_classes))
    for p, l in zip(predictions, labels):
        confusion_matrix[p][l] += 1
    return accuracy, confusion_matrix


def print_preds(predictions, y_batch, is_training=True):
    print(
        '\n ---------- ' + ('val' if not is_training else 'train') + ' predictions, {} predictions ----------\n'.format(
            str(predictions.shape[0])))
    for pi in range(min(10, predictions.shape[0])):
        pind = np.random.randint(0, predictions.shape[0])
        print(np.nonzero(predictions[pind]), np.nonzero(y_batch[pind]))


def evaluate_and_print(scores, labels, log_file=None, is_training=True, epoch=None, batch=None, loss=None):
    # assume inputs are np arrays
    # print('entered evaluate_and_print function')
    scores, labels = np.array(scores), np.array(labels)
    assert scores.shape == labels.shape, '[Error] scores.shape {}, labels.shape {} doesnot match'.format(scores.shape,
                                                                                                         labels.shape)
    predictions = get_prediction_sigmoid(scores)
    assert batch and is_training or not batch
    subset_acc = accuracy_score(labels, predictions)
    jaccard_sim = jaccard_similarity_score(labels, predictions)
    hamming_lo = hamming_loss(labels, predictions)
    if is_training:
        log_str = ("[Train]" if epoch is None and batch is None else
                   ("[Train - Epoch {}] ".format(epoch) if batch is None else
                    "---[train - epoch {}, batch {}] ".format(epoch, batch)))
        log_str += "{}: ".format(time.ctime().replace(' ', '_'))
        log_str += 'loss {:g},'.format(loss) if loss is not None else ''
        log_str += " subset_acc {:g}, jaccard_sim {:g}, hamming_lo {:g}".format(subset_acc, jaccard_sim, hamming_lo)
    else:
        log_str = "\n[Validation - Epoch {}] {}: subset_acc {:g}, jaccard_sim  {:g}, hamming_lo  {:g}\n".format(
            epoch, time.ctime().replace(' ', '_'), subset_acc, jaccard_sim, hamming_lo)
        print_preds(predictions, labels)

    print(log_str)
    if log_file:
        log_writer = open(log_file, 'a')
        log_writer.write(log_str + '\n')
        log_writer.close()


# ==================== Data Preparation ==============================
linesep('Loading data')

x, y, max_doc_len = preprocess_mimiciii.load_data(dataset_dir=FLAGS.dataset_dir, file_labels_fn=FLAGS.file_labels_fn,
                                     max_doc_len=FLAGS.max_doc_len)


linesep('Shuffle data')

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))

x_shuffled = []
for si in shuffle_indices:
    x_shuffled.append(x[si])  # [num_samples, string of max_doc_len of words]

y_shuffled = y[shuffle_indices]  # np.array(num_samples, num_classes)

n_dev_samples, n_test_samples = int(len(y) * 0.1), int(len(y) * 0.1)
n_train_samples = len(y) - n_dev_samples - n_test_samples

x_train, x_dev = x_shuffled[:n_train_samples], x_shuffled[n_train_samples:n_train_samples + n_dev_samples]
y_train, y_dev = y_shuffled[:n_train_samples], y_shuffled[n_train_samples:n_train_samples + n_dev_samples]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



linesep('string to int indices')

vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len, min_frequency=3)

x_train = np.array(list(vocab_processor.fit_transform(x_train)))  # x_train shape: (num_train_samples, max_doc_len)
x_dev = np.array(list(vocab_processor.transform(x_dev)))  # x_dev shape: (num_dev_samples, max_doc_len)

num_classes = y_train.shape[1]



log_file = FLAGS.task + \
           ('_small_' if '_small_' in FLAGS.file_labels_fn else '_') + \
           str(y_train.shape[1]) + 'labels_' + str(FLAGS.learning_rate) + \
           'lr_' + time.ctime().replace(' ', '_') + '.txt'
log_file = os.path.join(FLAGS.log_dir, log_file)
num_batches_per_epoch = int((n_train_samples - 1) / FLAGS.batch_size) + 1
evalute_per_batch = FLAGS.evaluate_per_batch if \
    (FLAGS.evaluate_per_batch and '_small_' not in FLAGS.file_labels_fn) \
    else num_batches_per_epoch * FLAGS.evaluate_per_epoch
decay_every_steps = 2000

assert len(vocab_processor.vocabulary_) > 100

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print('batch_size: {:d}'.format(FLAGS.batch_size))
print('x_train shape : {}'.format(x_train.shape))
print('x_val shape: {}'.format(x_dev.shape))
print('y_train shape: {}'.format(y_train.shape))
print('y_val shape: {}'.format(y_dev.shape))
print("Train/Dev split: {:d}/{:d}".format(n_train_samples, n_dev_samples))
print("num_classes: {:d}".format(y_train.shape[1]))
print("save features: {:d}".format(FLAGS.save_features))
print("save features per epoch: {:d}".format(FLAGS.save_features_per_epoch))
print("evaluate every batch: {:d}".format(evalute_per_batch))
print("checkpoint every epoch: {:d}".format(FLAGS.checkpoint_per_epoch))
print("number of filters: {}".format(FLAGS.num_filters))
print("num_batches_per_epoch : {}".format(num_batches_per_epoch))
print("learning rate: {}".format(FLAGS.learning_rate))
print("dropout keep prob: {}".format(FLAGS.dropout_keep_prob))
print("l2 reg lambda: {}".format(FLAGS.l2_reg_lambda))
print("learning rate decay: {}".format(FLAGS.learning_rate_decay))

# ====================== Training ============================
linesep('TRAINING')

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    log_writer = open(log_file, 'w')
    log_writer.close()

    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # cnn = TextCNNDeep(sequence_length=max_doc_len, num_classes=num_classes,
        #                   vocab_size=len(vocab_processor.vocabulary_),
        #                   embedding_size=FLAGS.embedding_dim,
        #                   kernel_sizes=[3],
        #                   l2_reg_lambda=FLAGS.l2_reg_lambda, multilabel=FLAGS.multilabel,
        #                   num_conv_layers=4, conv_num_filters=[128, 64, 32, 16], pooling_filter_size=3,
        #                   fc_sizes=[256, 64])

        cnn = TextCNN(sequence_length=max_doc_len, num_classes=num_classes,
                      vocab_size=len(vocab_processor.vocabulary_),
                      embedding_size=FLAGS.embedding_dim,
                      filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                      num_filters=FLAGS.num_filters, multilabel=FLAGS.multilabel,
                      pooling_filter_size=3, l2_reg_lambda=FLAGS.l2_reg_lambda)

        ##### Define Training procedure
        decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_every_steps,
                                                           FLAGS.learning_rate_decay,
                                                           staircase=True)  # decay around every 10 epoches
        optimizer = tf.train.AdamOptimizer(decayed_learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        ##### Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_{0}".format(
            'multilabel_' + str(num_classes) if FLAGS.multilabel else '1class'), time.ctime().replace(' ', '_')))
        print("Writing to {}".format(out_dir))
        print("Log file is {}\n".format(log_file))
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        ##### Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        ##### Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.load_model and FLAGS.load_model_folder:
            print('[INFO] loading model from ', tf.train.latest_checkpoint(FLAGS.load_model_folder))
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.load_model_folder))


        def train_step(x_batch, y_batch):
            # x_batch: np.array(batch_size, max_doc_len), y_batch: np.array(batch_size, num_classes)
            feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy, scores = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.scores], feed_dict)
            if FLAGS.multilabel and step % (num_batches_per_epoch // 100) == 0: evaluate_and_print(scores, y_batch,
                                                                                                   is_training=True,
                                                                                                   epoch=step // num_batches_per_epoch,
                                                                                                   batch=step % num_batches_per_epoch,
                                                                                                   loss=loss)
            return loss, accuracy, scores


        def dev_step(x_batch, y_batch):
            dev_size = len(x_batch)
            max_batch_size = 500
            num_batches = int((dev_size - 1) / max_batch_size) + 1
            acc, losses, dev_scores = [], [], []
            print("\n[Evaluation]{}\nNumber of batches in dev set is {}".format(time.ctime().replace(' ', '_'),
                                                                                (num_batches)))
            for i in range(num_batches):
                x_batch_dev, y_batch_dev = preprocess_mimiciii.get_batched(x_batch, y_batch, i * max_batch_size,
                                                                           min(dev_size, (i + 1) * max_batch_size))
                feed_dict = {cnn.input_x: x_batch_dev, cnn.input_y: y_batch_dev, cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy, scores = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.scores], feed_dict)
                dev_summary_writer.add_summary(summaries, step)
                acc.append(accuracy)
                losses.append(loss)
                dev_scores.extend(scores)
            print("Mean accuracy = {}, Mean loss = {}".format(np.mean(acc), np.mean(losses)))
            if FLAGS.multilabel: evaluate_and_print(dev_scores, y_batch, is_training=False,
                                                    epoch=step // num_batches_per_epoch, batch=None,
                                                    loss=np.mean(losses))


        train_scores, train_labels, train_loss, train_accuracy = [], [], [], []
        for x_batch, y_batch in preprocess_mimiciii.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs):
            loss, accuracy, scores = train_step(x_batch, y_batch)
            train_loss.append(loss)
            train_accuracy.append(accuracy)
            train_scores.extend(scores)
            train_labels.extend(y_batch)

            if tf.train.global_step(sess, global_step) % num_batches_per_epoch == 0:
                if FLAGS.multilabel: evaluate_and_print(train_scores, train_labels, is_training=True,
                                                        loss=np.mean(train_loss))
                train_scores, train_labels, train_loss, train_accuracy = [], [], [], []
                dev_step(x_dev, y_dev)

            if (tf.train.global_step(sess, global_step) + 1) % (
                        num_batches_per_epoch * FLAGS.checkpoint_per_epoch) == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=tf.train.global_step(sess, global_step))
