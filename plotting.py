import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, col, title, legend, xlabel, ylabel, filename):
    fig, (ax) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(x, y, color=col, label=legend, linewidth=3)
    ax.set_title(title, size=10)
    ax.set_xlabel(xlabel, size=8)
    ax.set_ylabel(ylabel, size=8)
    ax.grid()
    ax.legend()
    fig.savefig(filename)

def plot_return(x, y, col, title, legend, xlabel, ylabel, filename):
    fig, (ax) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(x, y, color=col, label=legend)
    ax.set_title(title, size=10)
    ax.set_xlabel(xlabel, size=8)
    ax.set_ylabel(ylabel, size=8)
    ax.grid()
    ax.legend()
    ax.set_ylim(-100, 25)
    fig.savefig(filename)

def plot_mult_return(xs, ys, title, legend, xlabel, ylabel, filename):
    fig, (ax) =  plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(xs[0], ys[0], label=legend[0])
    ax.plot(xs[1], ys[1], label=legend[1])
    ax.plot(xs[2], ys[2], label=legend[2])
    ax.set_title(title, size=10)
    ax.set_xlabel(xlabel, size=8)
    ax.set_ylabel(ylabel, size=8)
    ax.grid()
    ax.legend()
    ax.set_ylim(-140, 20)
    fig.savefig(filename)

def plot_sar(s, a, r, title, xlabel, filename):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    ax.plot(np.arange(0, len(s), 1), s, color='blue', label='state')
    ax.plot(np.arange(0, len(s), 1), a, color='red', label='action')
    ax.plot(np.arange(0, len(s), 1), r, color='green', label='reward')
    ax.set_title(title, size=10)
    ax.set_xlabel(xlabel, size=8)
    ax.grid()
    ax.legend()
    fig.savefig(filename)