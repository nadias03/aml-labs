import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def box_plot_a(method_name: str, scheme, rho, results, labels_name, pdf):
    plt.figure()

    plt.boxplot(results.values(), 
                    tick_labels=[f"{labels_val}" for labels_val in results.keys()])
    plt.xlabel(f"{labels_name} parameter value")
    plt.ylabel("accuracy")
    plt.title(f"Experiment 1 - {method_name} accuracy - scheme {scheme}, rho={rho}")

    pdf.savefig()
    plt.show()
    plt.close()

def box_plot_rho(method_name: str, scheme, a, results, labels_name, pdf):
    plt.figure()

    plt.boxplot(results.values(), 
                    tick_labels=[f"{labels_val}" for labels_val in results.keys()])
    plt.xlabel(f"{labels_name} parameter value")
    plt.ylabel("accuracy")
    plt.title(f"Experiment 2 - {method_name} accuracy - scheme {scheme}, a={a}")

    pdf.savefig()
    plt.show()
    plt.close()

def scatter_plot(X, y, a, rho, lda, qda, pdf):
    plt.figure(figsize=(12, 8))

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="blue",
        marker="o",
        label="class 0",
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="red",
        marker="x",
        label="class 1",
    )

    # grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    boundary_line_lda = lda.boundary_line(grid)
    boundary_line_qda = qda.boundary_line(grid)

    boundary_line_lda = boundary_line_lda.reshape(xx.shape)
    boundary_line_qda = boundary_line_qda.reshape(xx.shape)

    plt.contour(xx, yy, boundary_line_lda, levels=[0], colors="green")
    plt.contour(xx, yy, boundary_line_qda, levels=[0], colors="black")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="class 0",
            markerfacecolor="blue", markersize=8),
        Line2D([0], [0], marker="x", color="red", label="class 1",
            linestyle="None", markersize=8),
        Line2D([0], [0], color="green", lw=2, label="LDA boundary"),
        Line2D([0], [0], color="black", lw=2, label="QDA boundary"),
    ]

    plt.legend(handles=legend_elements)
    plt.title(f"Decision boundaries (a={a}, rho={rho})")
    plt.xlabel("x1")
    plt.ylabel("x2")

    pdf.savefig()
    plt.show()
    plt.close()

def box_plot_real_data(method_name: str, results, labels_name, data, pdf):
    plt.figure(figsize=(8, 4.8))

    plt.boxplot(results.values(), 
                    tick_labels=[f"{labels_val}" for labels_val in results.keys()])
    plt.xlabel(f"{labels_name} parameter value")
    plt.ylabel("accuracy")
    plt.title(f"Experiment 4 {data} - {method_name} accuracy for different train/test splits")

    pdf.savefig()
    plt.show()
    plt.close()