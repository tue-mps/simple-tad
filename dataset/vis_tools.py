from typing import Sequence, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from io import BytesIO
from sklearn.metrics import auc


def fig_to_cv2_image(fig):
    # Save the Matplotlib figure to a buffer in memory
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    [ax.cla() for ax in fig.get_axes()]
    plt.close(fig)
    return img


def threshold_curve_plots(
        x_values: Sequence[float],
        y_values: Sequence[float],
        thresholds: Sequence[float],
        x_label: str,
        y_label: str,
        plot_name: str,
        score: bool = False,
        to_img: bool = False,
        curve_type_correction: Optional[str] = None
):
    """
    Plot a threshold curve and optionally calculate the area under the curve (AUC).

    Parameters:
    - x_values: Sequence of x-axis values (e.g., recall or false positive rate)
    - y_values: Sequence of y-axis values (e.g., precision or true positive rate)
    - thresholds: Sequence of threshold values corresponding to x and y values
    - x_label: Label for the x-axis
    - y_label: Label for the y-axis
    - plot_name: Title of the plot
    - score: Boolean to indicate if AUC should be calculated and displayed (default is False)
    - to_img: Boolean to indicate if convert the resulting plot to image (default is False)
    - curve_type_correction: str to indicate correction type, 'roc' or 'pr'.
    """
    assert len(x_values) == len(y_values) == len(thresholds)

    # Ensure x_values are sorted and unique
    sorted_indices = np.argsort(x_values)
    x_values = np.array(x_values)[sorted_indices]
    y_values = np.array(y_values)[sorted_indices]
    thresholds = np.array(thresholds)[sorted_indices]

    # Remove duplicates
    unique_x_values, unique_indices = np.unique(x_values, return_index=True)
    y_values = y_values[unique_indices]
    thresholds = thresholds[unique_indices]
    x_values = unique_x_values

    if curve_type_correction is not None:
        if curve_type_correction == "roc":
            y0, y1 = 0., 1.
        elif curve_type_correction == "pr":
            y0, y1 = 1., 0.
        else:
            raise ValueError
        x_values = np.insert(x_values, 0, 0.)
        x_values = np.append(x_values, 1.)
        y_values = np.insert(y_values, 0, y0)
        y_values = np.append(y_values, y1)

    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    px = 1 / plt.rcParams['figure.dpi']
    figsize = (640 * px, 480 * px)

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    ax.plot(x_values, y_values, marker='o', linestyle='-', color='b')

    step = 5
    y = (10, -20)
    for i in range(0, len(thresholds), step):
        ax.plot(x_values[i], y_values[i], marker='o', linestyle='-', color='g')
        ax.annotate(f"{thresholds[i]:.2f}", (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, y[i % 2]),
                    ha='center')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1 / 2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1 / 2))
    ax.grid(which='major', linestyle='--', color='gray', alpha=0.5)
    ax.grid(which='minor', linestyle=':', color='gray', alpha=0.3)

    if score and x_values.shape[0] > 1:
        auc_score = auc(x_values, y_values)
        ax.text(0.40, 0.20, f'AUC: {auc_score:.2f}', transform=ax.transAxes, fontsize=12, ha='left')

    if to_img:
        fig = fig_to_cv2_image(fig)

    return fig
