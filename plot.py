import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def parse():
    p = argparse.ArgumentParser("Confusion Matrix Plotter")
    p.add_argument("folder", type=str, help="Pipeline folder")

    args = p.parse_args(None if sys.argv[1:] else ["-h"])
    args.folder = Path(args.folder)

    return args


def plot_confusion(folder: Path):
    fig = plt.figure(f"Confusion Matrix for {folder}")
    ax = fig.add_axes(111)

    expected_f = folder / "test.svm"
    confused_f = folder / "confusion.txt"

    for f in [expected_f, confused_f]:
        if not os.path.exists(f):
            print(f"Could not find {f}. Did you run the pipeline?")

    with open(confused_f, "r") as f:
        confused = [int(i) for i in f.readlines()]

    with open(expected_f, "r") as f:
        expected = [int(i[0]) for i in f.readlines()]

    assert len(confused) == len(expected)

    matrix = np.zeros((10, 10))
    for e, c in zip(expected, confused):
        matrix[e, c] += 1

    accuracy = matrix.trace() / matrix.sum()

    disp = ConfusionMatrixDisplay(matrix)

    disp.plot(ax=ax)
    plt.title(f"Accuracy: {accuracy*100:.2f}%")


def plot_scree(folder: Path):
    fig = plt.figure("Scree Plot")
    ax = fig.add_axes(111)

    eigenvalues = np.loadtxt(folder / "train.scree")
    eigenvalues = eigenvalues[::-1]

    ax.plot(eigenvalues, marker=".")
    ax.set(
        xlabel="Index",
        ylabel="Eigenvalue",
        title="PCA Scree",
    )


if __name__ == "__main__":
    args = parse()

    plot_confusion(args.folder)
    plot_scree(args.folder)

    plt.show()
