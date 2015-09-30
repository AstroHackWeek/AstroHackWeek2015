import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.externals.joblib import Memory
from .plot_2d_separator import plot_2d_separator

def make_handcrafted_dataset():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y


def plot_rbf_svm_parameters():
    X, y = make_handcrafted_dataset()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, C in zip(axes, [1e0, 5, 10, 100]):
        ax.scatter(X[:, 0], X[:, 1], s=150, c=np.array(['red', 'blue'])[y])

        svm = SVC(kernel='rbf', C=C).fit(X, y)
        plot_2d_separator(svm, X, ax=ax, eps=.5)
        ax.set_title("C = %f" % C)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    for ax, gamma in zip(axes, [0.1, .5, 1, 10]):
        ax.scatter(X[:, 0], X[:, 1], s=150, c=np.array(['red', 'blue'])[y])
        svm = SVC(gamma=gamma, kernel='rbf', C=1).fit(X, y)
        plot_2d_separator(svm, X, ax=ax, eps=.5)
        ax.set_title("gamma = %f" % gamma)


def plot_svm(log_C, log_gamma):
    X, y = make_handcrafted_dataset()
    C = 10. ** log_C
    gamma = 10. ** log_gamma
    svm = SVC(kernel='rbf', C=C, gamma=gamma).fit(X, y)
    ax = plt.gca()
    plot_2d_separator(svm, X, ax=ax, eps=.5)
    # plot data
    ax.scatter(X[:, 0], X[:, 1], s=150, c=np.array(['red', 'blue'])[y])
    # plot support vectors
    sv = svm.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=230, facecolors='none', zorder=10, linewidth=3)
    ax.set_title("C = %.4f gamma = %.4f" % (C, gamma))


def plot_svm_interactive():
    from IPython.html.widgets import interactive, FloatSlider
    C_slider = FloatSlider(min=-3, max=3, step=.1, value=0, readout=False)
    gamma_slider = FloatSlider(min=-2, max=2, step=.1, value=0, readout=False)
    return interactive(plot_svm, log_C=C_slider, log_gamma=gamma_slider)
