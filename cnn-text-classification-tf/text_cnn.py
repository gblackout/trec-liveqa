import tensorflow as tf
import numpy as np
import sys
import itertools


class TextCNN_V2(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, dense_size,
                 l2_coef, crf_lambda_doub, crf_lambda_cub, crf_lambda_quad, use_crf=True, init_w2v=None,
                 freez_w2v=False):
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
        self.input_wiki = tf.placeholder(tf.int32, [None, sequence_length], name="input_wiki")
        self.input_adm = tf.placeholder(tf.float32, [None, num_classes], name="input_adm")
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

        with tf.name_scope("concat_wiki"):
            self.input_wiki_x = tf.concat([self.input_wiki, self.input_x], axis=0) # (num_class+batch_size) X seq_len

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

            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_wiki_x)
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

        # ================= direct cosine similarity =================
        with tf.name_scope('split'):
            wiki_features = self.h_drop[:num_classes]
            ehr_features = self.h_drop[num_classes:]
            numOf_batch = tf.shape(ehr_features)[0]

        with tf.name_scope('dot_product'):
            feature_by_class = []
            for i in xrange(num_classes):
                wiki_mat = tf.tile(wiki_features[i], [numOf_batch])
                wiki_mat = tf.reshape(wiki_mat, [numOf_batch, num_filters_total])
                cosine_norm = tf.reduce_sum(tf.square(wiki_mat), axis=-1, keep_dims=True)
                cosine_norm = cosine_norm * tf.reduce_sum(tf.square(ehr_features), axis=-1, keep_dims=True)
                feature_by_class.append(tf.reduce_sum(ehr_features * wiki_mat / cosine_norm, axis=-1, keep_dims=True))
            self.scores = tf.concat(feature_by_class, axis=-1)

        # # ================= EHR concat wiki then dense =================
        #
        # # split wiki; concat adm; tile wiki; concat wiki by class
        # with tf.name_scope('split_concat_tile'):
        #     wiki_features = self.h_drop[:num_classes]
        #     ehr_features = self.h_drop[num_classes:]
        #     numOf_batch = tf.shape(ehr_features)[0]
        #
        #     ehr_adm_features = tf.concat([ehr_features, self.input_adm], axis=-1, name='concat_adm')
        #
        #     feature_by_class = []
        #     for i in xrange(num_classes):
        #         wiki_mat = tf.tile(wiki_features[i], [numOf_batch])
        #         wiki_mat = tf.reshape(wiki_mat, [numOf_batch, num_filters_total])
        #         feature_by_class.append(tf.concat([ehr_adm_features, wiki_mat], axis=-1, name='concat_wiki_%i' % i))
        #
        #     # (num_class X batch_size) X (num_filters_total+num_class+num_filters_total)
        #     self.all_features = tf.concat(feature_by_class, axis=0, name='concat_feature_by_class')
        #
        # # relu dense layer
        # with tf.name_scope("dense"):
        #     # TODO ad hoc
        #     W = tf.Variable(
        #         tf.truncated_normal(shape=[num_filters_total * 2 + num_classes, dense_size], stddev=0.1),
        #         name="W")
        #     self.l2_loss += tf.nn.l2_loss(W)
        #     b = tf.Variable(tf.constant(0.1, shape=[dense_size]), name="b")
        #
        #     self.scores = tf.nn.xw_plus_b(self.all_features, W, b, name="lin_transform")
        #     self.scores = tf.contrib.layers.batch_norm(self.scores, is_training=self.is_training)
        #     self.scores = tf.nn.relu(self.scores, name="relu")
        #
        # # linear output
        # with tf.name_scope("output"):
        #     output = []
        #     for i in xrange(num_classes):
        #         W = tf.Variable(tf.truncated_normal(shape=[dense_size, 1], stddev=0.1), name="W_%i" % i)
        #         self.l2_loss += tf.nn.l2_loss(W)
        #         b = tf.Variable(tf.constant(0.1, shape=[1]), name="b_%i" % i)
        #
        #         output.append(tf.nn.xw_plus_b(self.scores[numOf_batch * i:numOf_batch * (i + 1)], W,
        #                                       b))  # (num_class X batch_size)
        #
        #     self.scores = tf.concat(output, axis=-1)

        # # ================= EHR concat wiki-dot-EHR then dense =================
        #
        # # split wiki; concat adm; tile wiki; concat wiki by class
        # with tf.name_scope('split'):
        #     wiki_features = self.h_drop[:num_classes]
        #     ehr_features = self.h_drop[num_classes:]
        #     numOf_batch = tf.shape(ehr_features)[0]
        #
        # with tf.name_scope('dot_product'):
        #     feature_by_class = []
        #     for i in xrange(num_classes):
        #         wiki_mat = tf.tile(wiki_features[i], [numOf_batch])
        #         wiki_mat = tf.reshape(wiki_mat, [numOf_batch, num_filters_total])
        #         feature_by_class.append(tf.reduce_sum(ehr_features * wiki_mat, axis=-1, keep_dims=True))
        #     unnorm_distance = tf.concat(feature_by_class, axis=-1)
        #     distance = tf.nn.l2_normalize(unnorm_distance, dim=-1, name='norm_distant')
        #
        # with tf.name_scope('concat_all'):
        #     ehr_adm_features = tf.concat([ehr_features, self.input_adm], axis=-1, name='concat_adm')
        #     # batch_size X (num_filters_total+num_class+num_class)
        #     self.all_features = tf.concat([ehr_adm_features, distance], axis=-1, name='concat_dist')
        #
        # # relu dense layer
        # with tf.name_scope("dense"):
        #     # TODO ad hoc
        #     W = tf.Variable(tf.truncated_normal(shape=[num_filters_total + num_classes*2, dense_size], stddev=0.1),
        #                     name="W")
        #     self.l2_loss += tf.nn.l2_loss(W)
        #     b = tf.Variable(tf.constant(0.1, shape=[dense_size]), name="b")
        #
        #     dense_out = tf.nn.xw_plus_b(self.all_features, W, b, name="lin_transform")
        #     dense_out = tf.contrib.layers.batch_norm(dense_out, is_training=self.is_training)
        #     dense_out = tf.nn.relu(dense_out, name="relu")
        #
        # # linear output
        # with tf.name_scope("output"):
        #     W = tf.Variable(tf.truncated_normal(shape=[dense_size, num_classes], stddev=0.1), name="W")
        #     self.l2_loss += tf.nn.l2_loss(W)
        #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        #     self.scores = tf.nn.xw_plus_b(dense_out, W, b, name="lin_transform")

        # ======================================================================================================
        #                                                CRF
        # ======================================================================================================
        with tf.name_scope("CRF"):
            A = tf.Variable(tf.truncated_normal(shape=[num_classes, num_classes], stddev=0.1), name="A")
            A_no_diag = A * self.doub_diag_mask

            B = tf.Variable(tf.truncated_normal(shape=[num_classes, num_classes, num_classes], stddev=0.1), name="B")
            B_no_diag = B * self.cub_diag_mask

            C = tf.Variable(tf.truncated_normal(shape=[num_classes, num_classes,
                                                       num_classes, num_classes], stddev=0.1),name="C")
            C_no_diag = C * self.quad_diag_mask

            # ================= potential for all possible y =================
            # b X d * (k X d)^T = b X k
            phi_dot_all_y = tf.matmul(self.scores, self.all_y, transpose_b=True, name='phi_dot_all_y')
            doub_all_y = crf_lambda_doub * tf.reduce_sum(tf.matmul(self.all_y, A_no_diag) * self.all_y, axis=-1)  # k
            cub_all_y = crf_lambda_cub * tf.reduce_sum(
                                         tf.reduce_sum(
                                         tf.tensordot(self.all_y, B_no_diag, ([1], [0])) *
                                         tf.expand_dims(self.all_y, axis=-1),axis=-1) * self.all_y, axis=-1)
            quad_all_y = crf_lambda_quad * tf.reduce_sum(
                                           tf.reduce_sum(
                                           tf.reduce_sum(
                                           tf.tensordot(self.all_y, C_no_diag, ([1], [0])) *
                                           tf.expand_dims(tf.expand_dims(self.all_y, axis=-1), axis=-1), axis=-1) *
                                           tf.expand_dims(self.all_y, axis=-1), axis=-1) * self.all_y, axis=-1)

            # ================= potential for y in the batch =================
            phi_dot_train_y = tf.reduce_sum(self.scores * self.input_y, axis=-1, name='phi_dot_train_y')  # b
            doub_train_y = crf_lambda_doub * tf.reduce_sum(tf.matmul(self.input_y, A_no_diag) * self.input_y, axis=-1)
            cub_train_y = crf_lambda_cub * tf.reduce_sum(
                                           tf.reduce_sum(
                                           tf.tensordot(self.input_y, B_no_diag, ([1], [0])) *
                                           tf.expand_dims(self.input_y, axis=-1), axis=-1) * self.input_y, axis=-1)
            quad_train_y = crf_lambda_quad * tf.reduce_sum(
                                             tf.reduce_sum(
                                             tf.reduce_sum(
                                             tf.tensordot(self.input_y, C_no_diag, ([1], [0])) *
                                             tf.expand_dims(tf.expand_dims(self.input_y, axis=-1), axis=-1), axis=-1) *
                                             tf.expand_dims(self.input_y, axis=-1), axis=-1) * self.input_y, axis=-1)

            # ================= log parition function =================
            all_multi_potential = tf.add(doub_all_y, tf.add(cub_all_y, quad_all_y), name='all_multi_potential')
            all_loglikelihood = tf.add(phi_dot_all_y, all_multi_potential, name='all_loglikelihood')  # b X k
            # the value of log(Z(phi_i)) as b-dim vector
            log_Z = tf.reduce_logsumexp(all_loglikelihood, axis=-1, name='log_z')

            # ================= log likelihood =================
            train_multi_potential = tf.add(doub_train_y, tf.add(cub_train_y, quad_train_y),
                                           name='train_multi_potential')
            log_likelihood = tf.reduce_sum(phi_dot_train_y + train_multi_potential - log_Z,
                                           axis=-1, name='log_likelihood')

            self.unary_score = tf.reduce_sum(phi_dot_train_y - log_Z, axis=-1, name='unary_score')
            self.binary_score = tf.reduce_sum(doub_train_y - log_Z, axis=-1, name='binary_score')
            self.cubic_score = tf.reduce_sum(cub_train_y - log_Z, axis=-1, name='binary_score')
            self.quad_score = tf.reduce_sum(quad_train_y - log_Z, axis=-1, name='binary_score')

        # Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

            if use_crf:
                self.obj_loss = - log_likelihood
            else:
                self.obj_loss = tf.reduce_mean(losses)

            self.l2_loss = l2_coef * self.l2_loss
            self.loss = self.l2_loss + self.obj_loss

        # mean accuracy
        with tf.name_scope("accuracy"):
            if use_crf:
                inds = tf.argmax(all_loglikelihood, axis=-1, name='inds')
                inds = tf.cast(inds, tf.int32)
                iter_ind = tf.constant(0)
                preds = tf.constant([[-1.0] * num_classes])

                c = lambda iter_ind, preds: iter_ind < tf.shape(self.input_y)[0]
                b = lambda iter_ind, preds: [iter_ind + 1,
                                             tf.concat([preds, tf.expand_dims(self.all_y[inds[iter_ind], :], axis=0)],
                                                       axis=0)]
                r = tf.while_loop(c, b, loop_vars=[iter_ind, preds],
                                  shape_invariants=[iter_ind.get_shape(), tf.TensorShape([None, num_classes])])

                self.pred = r[1][1:, :]

                # TODO ad hoc
                self.cnn_pred = tf.nn.sigmoid(self.scores, name='no_round_preds')
                self.cnn_pred = tf.round(self.cnn_pred, name='preds')
                self.cnn_correct_pred = tf.cast(tf.equal(self.cnn_pred, tf.round(self.input_y)), tf.float32)
                self.cnn_accuracy = tf.reduce_mean(self.cnn_correct_pred)
            else:
                self.pred = tf.nn.sigmoid(self.scores, name='no_round_preds')
                # TODO ad hoc
                self.cnn_pred = None
                self.cnn_accuracy = None

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
