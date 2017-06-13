from __future__ import division
from __future__ import print_function

import preprocess_mimiciii
import collections
import math
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from os.path import join as joinpath
from utils.misc import get_output_folder, makedir

out_dir = get_output_folder('out', 'w2v')

log_dir = joinpath(out_dir, 'log')
summary_dir = joinpath(out_dir, 'summary')
checkpoint_dir = joinpath(out_dir, "checkpoint")

makedir(log_dir)
makedir(summary_dir)
makedir(checkpoint_dir)

data_loader = preprocess_mimiciii.DataLoader(0.9, 32)
data_loader.load_from_mat('../hyper_mat_data')
vocabulary_size = data_loader.vocab_size


X = []


def foo():
    for i in xrange(data_loader.X.shape[0]):
        yield data_loader.X[i, :]

for e in data_loader.vocab_processor.reverse(foo()):
    X += e


# Step 3: Function to generate a training batch for the skip-gram model.
data_index = 0
data = data_loader.X.flatten()


def generate_batch(batch_size, num_skips, skip_window):

    global data_index

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]

        for j in range(num_skips):

            while target in targets_to_avoid:
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)

    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], data_loader.vocab_processor.vocabulary_.reverse(batch[i]),
          '->', labels[i, 0], data_loader.vocab_processor.vocabulary_.reverse(labels[i, 0]))






# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()






# Step 5: Begin training.
num_steps = int(5e6)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    summary_writer = tf.summary.FileWriter(summary_dir, session.graph)
    train_loss_summary = tf.summary.scalar("test_mean_loss", loss)

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)

        feed_dict = {train_inputs: batch_inputs,
                     train_labels: batch_labels}

        _, loss_val, summaries = session.run([optimizer, loss, train_loss_summary], feed_dict=feed_dict)

        summary_writer.add_summary(summaries, step)

        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()

            for i in xrange(valid_size):
                valid_word = data_loader.vocab_processor.vocabulary_.reverse(valid_examples[i])
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word

                for k in xrange(top_k):
                    close_word = data_loader.vocab_processor.vocabulary_.reverse(nearest[k])
                    log_str = '%s %s,' % (log_str, close_word)

                print(log_str)

            saver.save(session, joinpath(checkpoint_dir, str(step) + '.ckpt'), global_step=step)

    final_embeddings = normalized_embeddings.eval()

    # step 5.5 output all embedding list for later use
    full_table = embeddings.eval()
    print('full table type ----> '+type(full_table))
    np.save(joinpath(log_dir, 'full_table'), full_table)







# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [data_loader.vocab_processor.vocabulary_.reverse(i) for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, filename=joinpath(log_dir, 'tsne.png'))

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
