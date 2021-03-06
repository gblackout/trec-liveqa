#! /usr/bin/env python
import math
import os
from os.path import join as joinpath
from progressbar import ProgressBar
from utils.misc import linesep, get_output_folder, makedir
from utils.visualization import prf_summary, output_summary, curr_prf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf, numpy as np
import preprocess_mimiciii
from text_cnn import TextCNN_V2


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


def main(FLAGS):
    # ==================== Data Preparation ==============================
    linesep('Loading data')
    data_loader = preprocess_mimiciii.DataLoader(0.9, FLAGS.batch_size)

    # Output directory for models and summaries
    out_name = 'multilabel_' + ('CRF_' if FLAGS.use_crf else '') + str(10)
    out_dir = get_output_folder(FLAGS.output_dir, out_name)
    makedir(out_dir)

    matdir = joinpath(out_dir, 'mat')
    makedir(matdir)

    # TODO ad hoc
    data_loader.load_from_text(FLAGS.file_labels_fn, FLAGS.stpwd_path,
                               [FLAGS.portion_threshold, FLAGS.length_threshold], matdir)

    num_batches_per_epoch = data_loader.compute_numOf_batch(data_loader.partition_ind, FLAGS.batch_size)

    log_path = joinpath(out_dir, FLAGS.log_path)
    summary_dir = joinpath(out_dir, 'summary')
    checkpoint_dir = joinpath(out_dir, "checkpoint")


    makedir(summary_dir)
    makedir(checkpoint_dir)

    # ====================== Training ============================
    linesep('compile model')

    sess = get_session(FLAGS)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    cnn = TextCNN_V2(sequence_length=data_loader.max_doc_len,
                     num_classes=data_loader.num_class,
                     vocab_size=data_loader.vocab_size,
                     embedding_size=FLAGS.embedding_dim,
                     filter_sizes=map(int, FLAGS.filter_sizes.split(',')),
                     num_filters=FLAGS.num_filters,
                     dense_size=FLAGS.dense_size,
                     l2_coef=FLAGS.l2_reg_lambda,
                     crf_lambda_doub=FLAGS.crf_lambda_doub,
                     crf_lambda_cub=FLAGS.crf_lambda_cub,
                     crf_lambda_quad=FLAGS.crf_lambda_quad,
                     use_crf=FLAGS.use_crf)


    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # training summary
    train_loss_summary = tf.summary.scalar("train_loss", cnn.loss)
    l2_loss_summary = tf.summary.scalar("l2_loss", cnn.l2_loss)
    score_summary = tf.summary.histogram("score_dist", cnn.scores)
    raw_pred_summary = tf.summary.histogram("train_raw_pred_dist", cnn.raw_pred)
    pred_summary = tf.summary.histogram("train_pred_dist", cnn.pred)

    train_summary_op = tf.summary.merge([train_loss_summary,
                                         l2_loss_summary,
                                         score_summary,
                                         pred_summary,
                                         raw_pred_summary
                                         ])

    # test summary
    test_mean_acc_pd = tf.placeholder(tf.float32, shape=None, name="test_mean_acc")
    test_weighted_f_pd = tf.placeholder(tf.float32, shape=None, name="test_weighted_f")
    test_loss_pd = tf.placeholder(tf.float32, shape=None, name="test_loss")
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
    test_loss_summary = tf.summary.scalar("test_loss", test_loss_pd)

    test_summary_op = tf.summary.merge([#test_acc_summary,
                                        test_weighted_f_summary,
                                        test_loss_summary,
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

    def evaluate(bestf1, d_loader, prf_hist, dist_hist):
        data_size = d_loader.X.shape[0] - d_loader.partition_ind
        num_batches = d_loader.compute_numOf_batch(data_size, FLAGS.batch_size)

        y_pred = np.zeros((data_size, d_loader.num_class), dtype=np.float32)

        cur_step = tf.train.global_step(sess, global_step)

        losses, acc_ls = [], []
        pbar = ProgressBar(maxval=num_batches).start()
        for x_batch, y_batch, batch_num, is_epochComplete in d_loader.batcher(train=False, batch_size=FLAGS.batch_size):
            feed_dict = {cnn.input_x: x_batch,
                         cnn.input_y: y_batch,
                         cnn.dropout_keep_prob: 1.0,
                         cnn.is_training: False}

            loss, test_acc, batch_pred = sess.run([cnn.loss, cnn.accuracy, cnn.pred], feed_dict)

            losses.append(loss)
            acc_ls.append(test_acc)
            chunk_end = min((batch_num + 1) * FLAGS.batch_size, data_size)
            y_pred[batch_num * FLAGS.batch_size:chunk_end, :] = batch_pred

            pbar.update(batch_num + 1)

            if is_epochComplete:
                break
        pbar.finish()

        y_true = d_loader.Y[d_loader.partition_ind:, :]

        prf_ls = prf(y_true, y_pred)
        prf_images, prf_hist = prf_summary(prf_hist, prf_ls)
        dist_image, dist_hist = output_summary(dist_hist, y_pred)
        curr_prf_image = curr_prf(d_loader.get_med_freq(), prf_ls)

        population = d_loader.get_med_freq()
        weights = population / np.sum(population)
        weighted_f = weights * prf_ls[2]
        weighted_f = np.sum(weighted_f)

        # whether this model performs best? if so save it
        if weighted_f > bestf1:
            bestf1 = weighted_f
            updatebest = True
        else:
            updatebest = False

        summaries = sess.run(test_summary_op,
                             {#test_mean_acc_pd: np.mean(acc_ls),
                              test_weighted_f_pd: weighted_f,
                              test_loss_pd: np.mean(losses),
                              test_precision_pd: np.expand_dims(prf_images[0], axis=0),
                              test_recall_pd: np.expand_dims(prf_images[1], axis=0),
                              test_fscore_pd: np.expand_dims(prf_images[2], axis=0),
                              test_dist_pd: np.expand_dims(dist_image, axis=0),
                              test_curr_prf_pd: np.expand_dims(curr_prf_image, axis=0)})

        summary_writer.add_summary(summaries, cur_step)

        # updated best summaries
        if updatebest:
            best_summaires = sess.run(best_test_summary_op,
                                      {test_weighted_f_pd: weighted_f,
                                       test_curr_prf_pd: np.expand_dims(curr_prf_image, axis=0)})
            summary_writer.add_summary(best_summaires, cur_step)

        return bestf1, prf_hist, dist_hist, updatebest

    linesep('initial model evaluate')
    best_weighted_f1, prf_hist, dist_hist, update_best = evaluate(best_weighted_f1, data_loader, prf_hist, dist_hist)

    epoch_cnt = 0
    eval_bystep = FLAGS.evaluate_freq > 0
    linesep('train epoch %i' % epoch_cnt)
    pbar = ProgressBar(maxval=num_batches_per_epoch).start()

    for x_batch, y_batch, batch_num, is_epochComplete in data_loader.batcher():

        feed_dict = {cnn.input_x: x_batch,
                     cnn.input_y: y_batch,
                     cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                     cnn.is_training: True}

        res = sess.run([train_op,
                        global_step,
                        train_summary_op,
                        ], feed_dict)

        cur_step, summaries = res[1:3]

        summary_writer.add_summary(summaries, cur_step)
        pbar.update(batch_num + 1)

        # eval model in step mode
        if eval_bystep and cur_step % FLAGS.evaluate_freq == 0:
            pbar.finish()

            linesep('evaluate model at step %i' % cur_step)
            best_weighted_f1, prf_hist, dist_hist, update_best = \
                evaluate(best_weighted_f1, data_loader, prf_hist, dist_hist)

            if update_best:
                saver.save(sess, joinpath(checkpoint_dir, 'best_model.ckpt'))

            pbar.start()
            pbar.update(batch_num + 1)

        if is_epochComplete:
            pbar.finish()

            # eval model in epoch mode
            if not eval_bystep and (epoch_cnt % (-FLAGS.evaluate_freq) == 0):
                linesep('evaluate model at epoch %i' % epoch_cnt)
                best_weighted_f1, prf_hist, dist_hist, update_best = \
                    evaluate(best_weighted_f1, data_loader, prf_hist, dist_hist)

                # if update_best:
                #     saver.save(sess, joinpath(checkpoint_dir, 'best_model.ckpt'))

            # save model
            if not eval_bystep and (epoch_cnt % (-FLAGS.evaluate_freq) == 0):
                linesep('save model at epoch %i' % epoch_cnt)
                saver.save(sess, joinpath(checkpoint_dir, 'model.ckpt'), global_step=cur_step)

            epoch_cnt += 1
            if epoch_cnt >= FLAGS.num_epochs:
                break

            linesep('train epoch %i' % epoch_cnt)
            pbar.start()

    return out_dir, best_weighted_f1


def bar(model_path):
    import json
    with open('filteredQuestion.json') as k:
        jn = json.load(k)
        X, rm = [], []

        for i, entry in enumerate(jn):
            text_one = entry['title'] + '. ' + entry['content']
            single_x, rv, _ = preprocess_mimiciii.DataLoader.parse_single(text_one, joinpath(model_path, 'mat'), 'stpwd')
            X.append(single_x)
            rm.append(np.array([rv]))
        X = np.concatenate(X, axis=0)

    return X, np.concatenate(rm, axis=0)


def test(input_filepath, output_filepath, model_path, FLAGS):

    # TODO the way of reading may change
    with open(input_filepath) as f:
        question = f.read().decode(encoding='utf-8', errors='ignore')
        question = question.strip()

    X1, rule_mask1, vocab_size = preprocess_mimiciii.DataLoader.parse_single(question, joinpath(model_path, 'mat'), 'stpwd')

    with tf.Session() as sess:

        cnn = TextCNN_V2(sequence_length=134,
                         num_classes=10,
                         vocab_size=vocab_size,
                         embedding_size=FLAGS.embedding_dim,
                         filter_sizes=map(int, FLAGS.filter_sizes.split(',')),
                         num_filters=FLAGS.num_filters,
                         dense_size=FLAGS.dense_size,
                         l2_coef=FLAGS.l2_reg_lambda,
                         crf_lambda_doub=FLAGS.crf_lambda_doub,
                         crf_lambda_cub=FLAGS.crf_lambda_cub,
                         crf_lambda_quad=FLAGS.crf_lambda_quad,
                         use_crf=FLAGS.use_crf)

        # new_saver = tf.train.import_meta_graph(joinpath(model_path, 'checkpoint/best_model.ckpt.meta'))
        new_saver = tf.train.Saver()
        new_saver.restore(sess, joinpath(model_path, 'checkpoint/model.ckpt-1785'))

        feed_dict = {
                     cnn.input_x: X1,
                     # cnn.input_x: X,
                     cnn.dropout_keep_prob: 1.0,
                     cnn.is_training: False}

        batch_pred = sess.run(cnn.pred, feed_dict)
        batch_pred_rule = np.clip(batch_pred + np.array([rule_mask1]), 0, 1)

        with open(output_filepath, 'w') as f:
            for e in batch_pred_rule[0].astype(int):
                print >> f, e,


if __name__ == '__main__':
    # ==================== Parameters ==============================

    # paths

    # TODO ad hoc
    tf.flags.DEFINE_string("file_labels_fn", 'type_data', "")

    tf.flags.DEFINE_string("stpwd_path", 'stpwd', "")
    tf.flags.DEFINE_string("matdata_dir", '../hyper_mat_data', "")
    tf.flags.DEFINE_string("w2v_path", '../full_table.npy', "")
    tf.flags.DEFINE_string("output_dir", 'out', "main output directory")
    tf.flags.DEFINE_string("log_path", 'log_file', "")
    tf.flags.DEFINE_integer("load_model", 0, "load model file")
    tf.flags.DEFINE_string("load_model_folder", './runs_sigmoid_1512/1494012307/checkpoints', "load model file")

    tf.flags.DEFINE_integer("multilabel", 1, "softmax or sigmoid")
    tf.flags.DEFINE_integer("portion_threshold", 0.99, "max number of words allowed in a document")
    tf.flags.DEFINE_integer("length_threshold", None, "max number of words allowed in a document")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("dense_size", 64, "size of the dense layer")
    # regularization
    tf.flags.DEFINE_float("dropout_keep_prob", 0.3, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularization lambda (default: 0.001)")
    # learning rate
    tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    # CRF
    tf.flags.DEFINE_float("crf_lambda_doub", 0.1, "")
    tf.flags.DEFINE_float("crf_lambda_cub", 0.01, "")
    tf.flags.DEFINE_float("crf_lambda_quad", 0.01, "")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
    # negative int -> every <evaluate_freq> epochs; positive int -> every <evaluate_freq> steps
    tf.flags.DEFINE_integer("evaluate_freq", -5, "Evaluate model every <evaluate_freq> steps/epoch for pos/neg input")
    tf.flags.DEFINE_integer("checkpoint_freq", 1, "Save model every <checkpoint_freq> epochs")
    tf.flags.DEFINE_integer("num_checkpoints", 100, "Number of checkpoints to store")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    tf.flags.DEFINE_boolean("use_crf", True, "")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--input", dest="INPUT", type="string", help="Number of samples to run")
    parser.add_option("--output", dest="OUTPUT", type="string", help="Top number of hypotheses to store")

    (options, args) = parser.parse_args()

    test(options.INPUT, options.OUTPUT, 'out/multilabel_CRF_12-run1', FLAGS)
    import sys
    sys.exit(0)

    # fine-tune hyper-parameters
    while True:

        FLAGS.num_epochs = 300
        FLAGS.num_checkpoints = 50

        param_list = [['num_epochs', FLAGS.num_epochs],
                      ['crf_lambda_doub', FLAGS.crf_lambda_doub],
                      ['crf_lambda_cub', FLAGS.crf_lambda_cub],
                      ['crf_lambda_quad', FLAGS.crf_lambda_quad],
                      ['portion_threshold', FLAGS.portion_threshold],
                      ['dropout_keep_prob', FLAGS.dropout_keep_prob],
                      ['dense_size', FLAGS.dense_size],
                      ['num_filters', FLAGS.num_filters]]

        linesep('Parameter')
        for name, v in param_list:
            print '%s%s:    %s' % (name, ' ' * (25 - len(name)), str(v))

        # start training
        out_dir, best_weighted_f1 = main(FLAGS)

        with open('performance_log', 'a') as f:
            print >> f, '%s\t%.4f\t%s' % (out_dir, best_weighted_f1, str(param_list))
