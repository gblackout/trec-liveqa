import tensorflow as tf
import numpy as np
import sys
import itertools


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 multilabel=0, pooling_filter_size=3, l2_reg_lambda=0.0, weighted_loss=0):


        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        if weighted_loss:
            self.sample_weight = tf.placeholder(tf.float32, [None, 1], name="sample_weight")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        # TODO Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                print('filter_size : {}, pooled : {}'.format(filter_size, pooled.get_shape()))

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print('h_pool_flat : {}'.format(self.h_pool_flat.get_shape()))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.nn.batch_normalization(self.scores, 0, 1, None, None, 0.01, name='bn')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Mean cross-entropy loss
        with tf.name_scope("loss"):
            if multilabel:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            else:
                if not weighted_loss:
                    losses = tf.losses.sparse_softmax_cross_entropy(logits=self.scores, labels=self.input_y)
                else:
                    losses = tf.losses.sparse_softmax_cross_entropy(logits=self.scores, labels=self.input_y, weights=self.sample_weight, scope=None)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class TextCNN_V2(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, dense_size,
                 l2_coef, crf_lambda, use_crf=True, init_w2v=None, freez_w2v=False):
        """
        init text cnn model
        
        input
        -----
        sequence_length: 
            maximum document length
        num_classes: 
            num of classes
        vocab_size: 
            vocabulary size
        embedding_size: 
            embedding size for each word        
        filter_sizes: 
            list of integers indicating the height of filter (width is the embedding size), where its 
            length determines the number of channels. E.g. [3,4,5] 
        num_filters:
            how many filters for each channel
        dense_size: 
            size of dense layer
        init_w2v:
            pretrained w2v numpy array
        """
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_adm = tf.placeholder(tf.float32, [None, num_classes], name="input_adm")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.population = tf.placeholder(tf.int32, shape=num_classes, name="input_x")

        # all possible tags: permute * num_classes
        self.all_y = np.array([seq for seq in itertools.product([0, 1], repeat=num_classes)], dtype=np.float32)
        self.numOf_permu = self.all_y.shape[0]
        self.all_y = tf.constant(self.all_y, name='all_y')

        # diag mask for multiplying A
        self.diag_mask = np.ones((num_classes, num_classes), dtype=np.float32)
        self.diag_mask[xrange(num_classes), xrange(num_classes)] = 0
        self.diag_mask = tf.constant(self.diag_mask, name='diag_mask')

        self.l2_loss = None

        # either learn a new one or load from a pretrained-one
        with tf.name_scope("embedding"):

            if type(init_w2v) == type(np.zeros(1)):
                assert init_w2v.shape[0] == vocab_size and init_w2v.shape[1] == embedding_size
                W = tf.Variable(init_w2v, name="W")
                if freez_w2v:
                    tf.stop_gradient(W)
                print '======>load word2vec from pretrained mat, freez_w2v is: %s' % str(freez_w2v)
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
                print '======>init random word2vec'

            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each channel
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                # adding L2 loss
                if self.l2_loss is None:
                    self.l2_loss = tf.nn.l2_loss(W)
                else:
                    self.l2_loss += tf.nn.l2_loss(W)

                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W,
                                    strides=[1, 1, 1, 1], padding="VALID", name="conv") + b
                # batch norm
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training)
                # Apply nonlinearity
                h = tf.nn.relu(conv, name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, -1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # TODO concate with adm encodings
        with tf.name_scope('concat'):
            self.h_drop = tf.concat([self.h_drop, self.input_adm], axis=-1)

        # relu dense layer
        with tf.name_scope("dense"):
            # TODO
            W = tf.Variable(tf.truncated_normal(shape=[num_filters_total+num_classes, dense_size], stddev=0.1), name="W")
            self.l2_loss += tf.nn.l2_loss(W)
            b = tf.Variable(tf.constant(0.1, shape=[dense_size]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="lin_transform")
            self.scores = tf.contrib.layers.batch_norm(self.scores, is_training=self.is_training)
            self.scores = tf.nn.relu(self.scores, name="relu")

        # linear output
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[dense_size, num_classes], stddev=0.1), name="W")
            self.l2_loss += tf.nn.l2_loss(W)
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.scores, W, b, name="lin_transform")

        # CRF
        with tf.name_scope("CRF"):
            A = tf.Variable(tf.truncated_normal(shape=[num_classes, num_classes], stddev=0.1), name="A")

            phi = tf.sigmoid(self.scores) # b X d
            A_no_diag = A * self.diag_mask  # d X d

            # b X d * (k X d)^T = b X k
            phi_dot_all_y = tf.matmul(phi, self.all_y, transpose_b=True, name='phi_dot_all_y')
            quad_all_y = crf_lambda * tf.reduce_sum(tf.matmul(self.all_y, A_no_diag) * self.all_y, axis=-1) # k

            phi_dot_train_y = tf.reduce_sum(phi * self.input_y, axis=-1, name='phi_dot_train_y') # b
            quad_train_y = crf_lambda * tf.reduce_sum(tf.matmul(self.input_y, A_no_diag) * self.input_y, axis=-1)  # b

            # the value of log(Z(phi_i)) as b-dim vector
            all_loglikelihood = tf.add(phi_dot_all_y, quad_all_y, name='all_loglikelihood') # b X k
            log_Z = tf.reduce_logsumexp(all_loglikelihood, axis=-1, name='log_z')

            log_likelihood = tf.reduce_sum(phi_dot_train_y + quad_train_y - log_Z, axis=-1, name='log_likelihood')


        # Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.obj_loss = tf.reduce_mean(losses)
            self.l2_loss = l2_coef * self.l2_loss
            if use_crf:
                self.loss = self.l2_loss - log_likelihood
            else:
                self.loss = self.l2_loss + self.obj_loss

        # mean accuracy
        with tf.name_scope("accuracy"):
            if use_crf:
                preds = []
                inds = tf.argmax(all_loglikelihood, axis=-1, name='inds')
                for i in xrange(tf.shape(inds)[0]):
                    preds.append(self.all_y[inds[i], :])
                self.pred = tf.stack(preds, name='no_round_preds')
            else:
                self.pred = tf.nn.sigmoid(self.scores, name='no_round_preds')

            self.pred = tf.round(self.pred, name='preds')
            self.correct_pred = tf.cast(tf.equal(self.pred, tf.round(self.input_y)), tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_pred)

        # # class-wise precision, recall, F1
        # with tf.name_scope("PRF"):
        #     y_intersect = tf.reduce_sum(self.correct_pred, axis=0) # num_class-dim vector
        #     y_pred = tf.cast(tf.reduce_sum(self.pred, axis=0), tf.float32) + 0.001
        #     y_true = tf.reduce_sum(self.input_y, axis=0) + 0.001
        #
        #     self.p = y_intersect / y_pred
        #     self.r = y_intersect / y_true
        #     self.f = 2 * self.p * self.r / (self.p+self.r)
        #
        #     self.weighted_f = tf.reduce_sum(self.f / (self.population / tf.reduce_sum(self.population)))

# if __name__ == '__main__':
#     lr =0.01
#
#     beta2 = 0.999
#     beta1 = 0.9
#
#     t_ls = np.arange(0, int(1e6), dtype=np.float32)
#     y_ls = []
#     for t in t_ls:
#         y_ls.append(lr * np.sqrt(1 - beta2**t) / (1 - beta1**t))
#
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#
#     plt.plot(t_ls, y_ls)
#     plt.show()