import matplotlib
import sys
if sys.platform.startswith('linux'):
    matplotlib.use('Agg')
elif sys.platform.startswith('darwin'):
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm


def fig2data(fig):
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    buf = buf.reshape(h, w, 3)

    # # Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis=2)
    return buf


def surface_mat(Z):

    numOf_class = Z.shape[1]
    n_samples = Z.shape[0]

    fig = plt.figure(figsize=(10, 7.5), dpi=100)
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, numOf_class, 1)
    Y = np.arange(0, n_samples, 1)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0.005, edgecolors='k', antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=28., azim=56)
    plt.tight_layout()

    mat_fig = fig2data(fig)
    plt.close(fig)

    return mat_fig


def prf_summary(prev, prf):
    if prev is not None:
        for i in xrange(3):
            prev[i] = np.append(prev[i], np.expand_dims(prf[i], axis=0), axis=0)
    else:
        prev = [np.expand_dims(e, axis=0) for e in prf]

    return [surface_mat(prev[i]) for i in xrange(3)], prev


def output_summary(prev, y_pred):
    sum_pred = np.sum(y_pred, axis=0)
    norm_const = np.sum(sum_pred)
    sum_pred = sum_pred / norm_const if norm_const != 0 else sum_pred
    if prev is not None:
        prev = np.append(prev, np.expand_dims(sum_pred, axis=0), axis=0)
    else:
        prev = np.expand_dims(sum_pred, axis=0)

    return surface_mat(prev), prev


def curr_prf(population, prf):

    x = np.arange(population.shape[0])
    inds = np.argsort(-population)

    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.plot(x, prf[0][inds], 'r--o', linewidth=2, label='precision', markersize=8)
    ax1.plot(x, prf[1][inds], 'r:^', linewidth=2, label='recall', markersize=9)
    ax1.plot(x, prf[2][inds], 'k-', linewidth=3, label='F1')
    ax1.set_xlabel('med index')
    plt.xticks(x, x[inds])
    handles, _ = ax1.get_legend_handles_labels()
    plt.legend(handles, ['precision', 'recall', 'F1'])

    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('PRF', color='r')
    ax1.tick_params('y', colors='r')
    ax1.set_ylim([0, 1])
    ax1.xaxis.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, population[inds], 'b:')
    ax2.set_ylabel('med frequency', color='b')
    ax2.tick_params('y', colors='b')

    fig.tight_layout()
    mat_fig = fig2data(fig)
    plt.close(fig)

    return mat_fig


if __name__ == '__main__':
    # plot_surface(np.random.rand(20, 50)+(np.array([range(20)]).T)*0.1)
    from PIL import Image

    # Image.fromarray(surface_mat(np.random.rand(20, 50)-(np.array([range(20)]).T)*0.1)).show()

    Image.fromarray(curr_prf(np.random.rand(50)*50, [np.random.rand(50) for _ in xrange(3)])).show()