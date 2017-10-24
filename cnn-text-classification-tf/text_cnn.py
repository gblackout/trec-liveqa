import tensorflow as tf
import numpy as np
import sys
import itertools


class TextCNN_V2(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, dense_size,
                 l2_coef, crf_lambda_doub, crf_lambda_cub, crf_lambda_quad, use_crf=True):
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
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.population = tf.placeholder(tf.int32, shape=num_classes, name="population")

        # all possible tags: permute * num_classes
        self.all_y = np.array([seq for seq in itertools.product([0, 1], repeat=num_classes)], dtype=np.float32)
        self.numOf_permu = self.all_y.shape[0]
        self.all_y = tf.constant(self.all_y, name='all_y')

        # diag mask for multiplying A, B and C
        self.doub_diag_mask = TextCNN_V2.get_mask_tensor((num_classes, num_classes))
        self.cub_diag_mask = TextCNN_V2.get_mask_tensor((num_classes, num_classes, num_classes))
        self.quad_diag_mask = TextCNN_V2.get_mask_tensor((num_classes, num_classes, num_classes, num_classes))

        self.doub_diag_mask = tf.constant(self.doub_diag_mask, name='doub_diag_mask')
        self.cub_diag_mask = tf.constant(self.cub_diag_mask, name='cub_diag_mask')
        self.quad_diag_mask = tf.constant(self.quad_diag_mask, name='quad_diag_mask')

        self.l2_loss = None

        # either learn a new one or load from a pretrained-one
        with tf.name_scope("embedding"):
            print '======>init random word2vec'
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")

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

        # relu dense layer
        with tf.name_scope("dense"):
            W = tf.Variable(tf.truncated_normal(shape=[num_filters_total, dense_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[dense_size]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)

            dense_out = tf.nn.xw_plus_b(self.h_drop, W, b, name="lin_transform")
            dense_out = tf.contrib.layers.batch_norm(dense_out, is_training=self.is_training)
            dense_out = tf.nn.relu(dense_out, name="relu")

        # linear output
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[dense_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)

            self.scores = tf.nn.xw_plus_b(dense_out, W, b, name="lin_transform")

        # Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

            self.obj_loss = tf.reduce_mean(losses)

            self.l2_loss = l2_coef * self.l2_loss
            self.loss = self.l2_loss + self.obj_loss

        # mean accuracy
        with tf.name_scope("accuracy"):

            self.raw_pred = tf.nn.sigmoid(self.scores, name='no_round_preds')

            self.pred = tf.round(self.raw_pred, name='prediction')
            self.correct_pred = tf.cast(tf.equal(self.pred, tf.round(self.input_y)), tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_pred)

    @staticmethod
    def get_mask_tensor(dim):
        mask = np.ones(dim, dtype=np.float32)
        numOf_dim = len(dim)
        numOf_class = dim[0]

        if numOf_dim == 2:
            inds = itertools.product(range(numOf_class), range(numOf_class))
        elif numOf_dim == 3:
            inds = itertools.product(range(numOf_class), range(numOf_class), range(numOf_class))
        elif numOf_dim == 4:
            inds = itertools.product(range(numOf_class), range(numOf_class), range(numOf_class), range(numOf_class))
        else:
            raise ValueError

        for ind in inds:
            mask[ind] = 0.0

        return mask
