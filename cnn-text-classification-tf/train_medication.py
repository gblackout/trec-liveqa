#! /usr/bin/env python
import math
import os
from os.path import join as joinpath
from progressbar import ProgressBar
from utils.misc import linesep, get_output_folder, makedir
from utils.visualization import prf_summary, output_summary, curr_prf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf, numpy as np
from sklearn.metrics import jaccard_similarity_score
import preprocess_mimiciii
from text_cnn import TextCNN, TextCNN_V2


def prf(y_true, y_pred):
    """
    input
    -----
    y_true:
        n_samples * n_class binary mat
    y_pred:
        n_samples * n_class binary mat
    output
    -----
        3 * n_class float mat: 1st row is precision, 2nd recall and 3rd f1
    """
    numOf_class = y_true.shape[1]

    hits = np.sum(np.logical_and(y_true > 0, y_pred > 0), axis=0)
    true_sum = y_true.sum(axis=0)
    pred_sum = y_pred.sum(axis=0)

    res = [np.zeros(numOf_class, dtype=np.float32) for _ in xrange(3)]
    for i in xrange(numOf_class):
        res[0][i] = p = hits[i] / pred_sum[i] if pred_sum[i] != 0 else 0
        res[1][i] = r = hits[i] / true_sum[i] if true_sum[i] != 0 else 0
        res[2][i] = 2 * p * r / (p + r) if (p + r) != 0 else 0

    return res


def get_prediction_sigmoid(scores, threshold=0.5):
    predictions = np.zeros_like(scores)
    for row, logits_v in enumerate(scores):
        for col, value in enumerate(logits_v):
            if math.exp(-np.logaddexp(0, -value)) >= threshold:
                predictions[row][col] = 1
            else:
                predictions[row][col] = 0

    return predictions


def get_session(args, shared=True):
    if shared:
        # since GPUs are shared we want soft placement if it is in use
        session_conf = tf.ConfigProto(allow_soft_placement=args.allow_soft_placement,
                                      log_device_placement=args.log_device_placement)
        # allocate mem when needed, since GPUs are shared
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
    else:
        session = tf.Session()

    return session


if __name__ == '__main__':
    # ==================== Parameters ==============================

    # paths

    # TODO ad hoc
    tf.flags.DEFINE_string("file_labels_fn", '../hyper_label_index_withadm', "")

    tf.flags.DEFINE_string("stpwd_path", '../stpwd', "")
    tf.flags.DEFINE_string("dataset_dir", '../uni_containers_tmp', "")
    tf.flags.DEFINE_string("matdata_dir", '../hyper_mat_data', "")
    tf.flags.DEFINE_string("w2v_path", '../full_table.npy', "")
    tf.flags.DEFINE_string("output_dir", 'out', "main output directory")
    tf.flags.DEFINE_string("log_path", 'log_file', "")
    tf.flags.DEFINE_integer("load_model", 0, "load model file")
    tf.flags.DEFINE_string("load_model_folder", './runs_sigmoid_1512/1494012307/checkpoints', "load model file")

    tf.flags.DEFINE_integer("multilabel", 1, "softmax or sigmoid")
    tf.flags.DEFINE_integer("max_doc_len", None, "max number of words allowed in a document")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("dense_size", 64, "size of the dense layer")
    # regularization
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.001)")
    # learning rate
    tf.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
    tf.flags.DEFINE_float("learning_rate_decay", 0.9, "learning rate decay")
    tf.flags.DEFINE_float("decay_every_steps", 2000, "decay_every_steps")
    # CRF
    tf.flags.DEFINE_float("crf_lambda", 0.5, "")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 24, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
    # negative int -> every <evaluate_freq> epochs; positive int -> every <evaluate_freq> steps
    tf.flags.DEFINE_integer("evaluate_freq", -1, "Evaluate model every <evaluate_freq> steps/epoch for pos/neg input")
    tf.flags.DEFINE_integer("checkpoint_freq", 1, "Save model every <checkpoint_freq> epochs")
    tf.flags.DEFINE_integer("num_checkpoints", 100, "Number of checkpoints to store")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    tf.flags.DEFINE_boolean("freez_w2v", True, "")
    tf.flags.DEFINE_boolean("use_crf", True, "")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    linesep('Parameter')
    for attr, value in sorted(FLAGS.__flags.items()):
        print '%s%s:    %s' % (attr.upper(), ' ' * (25 - len(attr)), str(value))

    # ==================== Data Preparation ==============================
    linesep('Loading data')
    data_loader = preprocess_mimiciii.DataLoader(0.9, FLAGS.batch_size)

    # TODO ad hoc
    data_loader.load_from_text(FLAGS.dataset_dir, FLAGS.file_labels_fn, FLAGS.stpwd_path,
                               max_doc_len=FLAGS.max_doc_len)

    # try:
    #     data_loader.load_from_mat(FLAGS.matdata_dir)
    #     print '======>found mat data in %s' % FLAGS.matdata_dir
    # except:
    #     print '======>mat data not found in %s loading from container in ' % FLAGS.matdata_dir
    #     data_loader.load_from_text(FLAGS.dataset_dir, FLAGS.file_labels_fn, FLAGS.stpwd_path,
    #                                max_doc_len=FLAGS.max_doc_len)
    #     print '======>saving mat data to %s' %FLAGS.matdata_dir
    #     data_loader.save_mat(FLAGS.matdata_dir)


    num_batches_per_epoch = data_loader.compute_numOf_batch(data_loader.partition_ind, FLAGS.batch_size)

    # Output directory for models and summaries
    out_name = 'multilabel_' + ('CRF_' if FLAGS.use_crf else '') + str(data_loader.num_class)
    out_dir = get_output_folder(FLAGS.output_dir, out_name)
    log_path = joinpath(out_dir, FLAGS.log_path)
    summary_dir = joinpath(out_dir, 'summary')
    checkpoint_dir = joinpath(out_dir, "checkpoint")

    makedir(out_dir)
    makedir(summary_dir)
    makedir(checkpoint_dir)

    # ====================== Training ============================
    linesep('compile model')

    sess = get_session(FLAGS)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # cnn = TextCNN(sequence_length=data_loader.max_doc_len, num_classes=data_loader.num_class,
    #               vocab_size=data_loader.vocab_size,
    #               embedding_size=FLAGS.embedding_dim,
    #               filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
    #               num_filters=FLAGS.num_filters, multilabel=FLAGS.multilabel,
    #               pooling_filter_size=3, l2_reg_lambda=FLAGS.l2_reg_lambda)

    cnn = TextCNN_V2(sequence_length=data_loader.max_doc_len,
                     num_classes=data_loader.num_class,
                     vocab_size=data_loader.vocab_size,
                     embedding_size=FLAGS.embedding_dim,
                     filter_sizes=map(int, FLAGS.filter_sizes.split(',')),
                     num_filters=FLAGS.num_filters,
                     dense_size=FLAGS.dense_size,
                     l2_coef=FLAGS.l2_reg_lambda,
                     crf_lambda=FLAGS.crf_lambda,
                     use_crf=FLAGS.use_crf,
                     init_w2v=None,
                     freez_w2v=FLAGS.freez_w2v)

    # define Training procedure
    # decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_every_steps,
    #                                                    FLAGS.learning_rate_decay,
    #                                                    staircase=True)  # decay around every 10 epoches
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # training summary
    train_loss_summary = tf.summary.scalar("train_loss", cnn.loss)
    l2_loss_summary = tf.summary.scalar("l2_loss", cnn.l2_loss)
    # train_acc_summary = tf.summary.scalar("train_accuracy", cnn.accuracy)
    # train_u_score_summary = tf.summary.scalar("unary_score", cnn.unary_score)
    # train_bi_score_summary = tf.summary.scalar("binary_score", cnn.binary_score)
    # train_cnn_acc_summary = tf.summary.scalar("train_cnn_accuracy", cnn.cnn_accuracy)

    train_summary_op = tf.summary.merge([train_loss_summary,
                                         l2_loss_summary,
                                         # train_acc_summary,
                                         # train_u_score_summary,
                                         # train_bi_score_summary,
                                         # train_cnn_acc_summary
                                         ])

    # test summary
    test_mean_acc_pd = tf.placeholder(tf.float32, shape=None, name="test_mean_acc")
    test_weighted_f_pd = tf.placeholder(tf.float32, shape=None, name="test_weighted_f")
    test_cnn_weighted_f_pd = tf.placeholder(tf.float32, shape=None, name="test_cnn_weighted_f")
    # images
    test_precision_pd = tf.placeholder(tf.uint8, shape=None, name='test_precision')
    test_recall_pd = tf.placeholder(tf.uint8, shape=None, name='test_recall')
    test_fscore_pd = tf.placeholder(tf.uint8, shape=None, name='test_fscore')
    test_dist_pd = tf.placeholder(tf.uint8, shape=None, name='test_distribution')
    test_curr_prf_pd = tf.placeholder(tf.uint8, shape=None, name='test_curr_prf')
    # image history arrays
    prf_hist = None
    dist_hist = None
    # record the best score on TEST set
    best_weighted_f1 = 0.0

    test_acc_summary = tf.summary.scalar("test_mean_acc", test_mean_acc_pd)
    test_weighted_f_summary = tf.summary.scalar("test_weighted_f", test_weighted_f_pd)
    # test_cnn_weighted_f_summary = tf.summary.scalar("test_cnn_weighted_f", test_cnn_weighted_f_pd)
    test_precision_summary = tf.summary.image('test_precision', test_precision_pd)
    test_recall_summary = tf.summary.image('test_recall', test_recall_pd)
    test_fscore_summary = tf.summary.image('test_fscore', test_fscore_pd)
    test_dist_summary = tf.summary.image('test_distribution', test_dist_pd)
    test_curr_prf_summary = tf.summary.image('test_curr_prf', test_curr_prf_pd)

    test_summary_op = tf.summary.merge([test_acc_summary,
                                        test_weighted_f_summary,
                                        # test_cnn_weighted_f_summary,
                                        test_precision_summary,
                                        test_recall_summary,
                                        test_fscore_summary,
                                        test_dist_summary,
                                        test_curr_prf_summary
                                        ])

    # best summaries
    best_test_weighted_f_summary = tf.summary.scalar("best_test_weighted_f", test_weighted_f_pd)
    best_test_curr_prf_summary = tf.summary.image('best_test_curr_prf', test_curr_prf_pd)

    best_test_summary_op = tf.summary.merge([best_test_weighted_f_summary, best_test_curr_prf_summary])

    # init writer and saver
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # initialize all variables
    sess.run(tf.global_variables_initializer())
    if FLAGS.load_model and FLAGS.load_model_folder:
        print('[INFO] loading model from ', tf.train.latest_checkpoint(FLAGS.load_model_folder))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.load_model_folder))


    def evaluate(d_loader, prf_hist, dist_hist):
        global best_weighted_f1
        data_size = d_loader.X.shape[0] - d_loader.partition_ind
        num_batches = d_loader.compute_numOf_batch(data_size, FLAGS.batch_size)

        y_pred = np.zeros((data_size, d_loader.num_class), dtype=np.float32)
        # TODO ad hoc
        cnn_pred = np.zeros((data_size, d_loader.num_class), dtype=np.float32)

        cur_step = tf.train.global_step(sess, global_step)

        losses, acc_ls = [], []
        pbar = ProgressBar(maxval=num_batches).start()
        for x_batch, y_batch, adm_batch, batch_num, is_epochComplete in d_loader.batcher(train=False,
                                                                                         batch_size=FLAGS.batch_size):
            feed_dict = {cnn.input_x: x_batch,
                         cnn.input_y: y_batch,
                         cnn.input_adm: adm_batch,
                         cnn.dropout_keep_prob: 1.0,
                         cnn.is_training: False}

            loss, test_acc, batch_pred, batch_cnn_pred = sess.run([cnn.loss,
                                                                   cnn.accuracy,
                                                                   cnn.pred,
                                                                   cnn.cnn_pred], feed_dict)

            losses.append(loss)
            acc_ls.append(test_acc)
            chunk_end = min((batch_num + 1) * FLAGS.batch_size, data_size)
            y_pred[batch_num * FLAGS.batch_size:chunk_end, :] = batch_pred
            cnn_pred[batch_num * FLAGS.batch_size:chunk_end, :] = batch_cnn_pred

            pbar.update(batch_num + 1)

            if is_epochComplete:
                break
        pbar.finish()

        y_true = d_loader.Y[d_loader.partition_ind:, :]

        jaccard_sim = jaccard_similarity_score(y_true, y_pred)
        prf_ls = prf(y_true, y_pred)
        prf_images, prf_hist = prf_summary(prf_hist, prf_ls)
        dist_image, dist_hist = output_summary(dist_hist, y_pred)
        curr_prf_image = curr_prf(d_loader.get_med_freq(), prf_ls)

        population = d_loader.get_med_freq()
        weights = population / np.sum(population)
        weighted_f = weights * prf_ls[2]
        weighted_f = np.sum(weighted_f)

        cnn_prf_ls = prf(y_true, cnn_pred)
        cnn_weighted_f = weights * cnn_prf_ls[2]
        cnn_weighted_f = np.sum(cnn_weighted_f)

        # whether this model performs best? if so save it
        if weighted_f > best_weighted_f1:
            best_weighted_f1 = weighted_f
            update_best = True
        else:
            update_best = False

        summaries = sess.run(test_summary_op,
                                {test_mean_acc_pd: np.mean(acc_ls),
                                 test_weighted_f_pd: weighted_f,
                                 test_cnn_weighted_f_pd: cnn_weighted_f,
                                 test_precision_pd: np.expand_dims(prf_images[0], axis=0),
                                 test_recall_pd: np.expand_dims(prf_images[1], axis=0),
                                 test_fscore_pd: np.expand_dims(prf_images[2], axis=0),
                                 test_dist_pd: np.expand_dims(dist_image, axis=0),
                                 test_curr_prf_pd: np.expand_dims(curr_prf_image, axis=0)})

        summary_writer.add_summary(summaries, cur_step)

        # updated best summaries
        if update_best:
            best_summaires = sess.run(best_test_summary_op,
                                      {test_weighted_f_pd: weighted_f,
                                       test_curr_prf_pd: np.expand_dims(curr_prf_image, axis=0)})
            summary_writer.add_summary(best_summaires, cur_step)

        return prf_hist, dist_hist, update_best


    linesep('initial model evaluate')
    prf_hist, dist_hist, update_best = evaluate(data_loader, prf_hist, dist_hist)

    epoch_cnt = 0
    eval_bystep = FLAGS.evaluate_freq > 0
    linesep('train epoch %i' % epoch_cnt)
    pbar = ProgressBar(maxval=num_batches_per_epoch).start()

    for x_batch, y_batch, adm_batch, batch_num, is_epochComplete in data_loader.batcher():

        feed_dict = {cnn.input_x: x_batch,
                     cnn.input_y: y_batch,
                     cnn.input_adm: adm_batch,
                     cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                     cnn.is_training: True}

        res = sess.run([train_op,
                        global_step,
                        train_summary_op,
                        cnn.loss,
                        cnn.scores,
                        # cnn.accuracy,
                        # cnn.unary_score,
                        # cnn.binary_score,
                        # cnn.cnn_accuracy
                        ], feed_dict)

        cur_step, summaries = res[1:3]

        summary_writer.add_summary(summaries, cur_step)
        pbar.update(batch_num + 1)

        # eval model in step mode
        if eval_bystep and cur_step % FLAGS.evaluate_freq == 0:
            pbar.finish()

            linesep('evaluate model at step %i' % cur_step)
            prf_hist, dist_hist, update_best = evaluate(data_loader, prf_hist, dist_hist)

            if update_best:
                saver.save(sess, joinpath(checkpoint_dir, 'best_model.ckpt'))

            pbar.start()
            pbar.update(batch_num + 1)

        if is_epochComplete:
            pbar.finish()

            # eval model in epoch mode
            if not eval_bystep and (epoch_cnt % (-FLAGS.evaluate_freq) == 0):
                linesep('evaluate model at epoch %i' % epoch_cnt)
                prf_hist, dist_hist, update_best = evaluate(data_loader, prf_hist, dist_hist)

                if update_best:
                    saver.save(sess, joinpath(checkpoint_dir, 'best_model.ckpt'))

            # save model
            if epoch_cnt % FLAGS.checkpoint_freq == 0:
                linesep('save model at epoch %i' % epoch_cnt)
                saver.save(sess, joinpath(checkpoint_dir, 'model.ckpt'), global_step=cur_step)

            epoch_cnt += 1
            if epoch_cnt >= FLAGS.num_epochs:
                break

            linesep('train epoch %i' % epoch_cnt)
            pbar.start()
