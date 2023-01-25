from typing import Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns


# colour definition
black = np.array([0, 0, 0, 255])
white = np.array([255, 255, 255, 255])
yellow = np.array([255, 255, 0, 255])
red = np.array([255, 0, 0, 255])
blue = np.array([0, 0, 255, 255])
light_blue = np.array([0, 255, 255, 255])
green = np.array([0, 255, 0, 255])
pink = np.array([255, 0, 255, 255])


# function which trasform mask values 0,1,2,3,4,5,6,7 into colours
def num_to_colors(mask, height, width):
    col_mask = np.zeros((height, width, 4))
    for j in range(0, width):
        for i in range(0, height):
            if mask[i, j] == 0:
                col_mask[i, j] = black
            elif mask[i, j] == 1:
                col_mask[i, j] = red
            elif mask[i, j] == 2:
                col_mask[i, j] = yellow
            elif mask[i, j] == 3:
                col_mask[i, j] = white
            elif mask[i, j] == 4:
                col_mask[i, j] = blue
            elif mask[i, j] == 5:
                col_mask[i, j] = light_blue
            elif mask[i, j] == 6:
                col_mask[i, j] = pink
            elif mask[i, j] == 7:
                col_mask[i, j] = green

    return col_mask


def train_test_validate_split(
    X_data, y_data, train_ratio=0.75, validation_ration=0.15, test_ratio=0.1
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Take X, y, use default ratios of train/validate/test of 0.75. 0.15, 0.1
    """
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=1 - train_ratio
    )

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio)
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def calculate_f1_micro_macro(test_sequence, predictions) -> None:
    """
    Takes a Sequence object and array of predictions

    Prints Macro and Micro F1 scores
    """
    masks = get_masks_from_keras_sequence(test_sequence)

    f1_macro = []
    f1_micro = []
    for mask, prediction in zip(masks, predictions):
        f1_macro.append(f1_score(mask.flatten(), prediction.flatten(), average="macro"))
        f1_micro.append(f1_score(mask.flatten(), prediction.flatten(), average="micro"))

    print(f"Macro F1: {np.average(f1_macro)}")
    print(f"Micro F1: {np.average(f1_micro)}")


def get_masks_from_keras_sequence(test_sequence) -> np.array:
    """
    Take a keras.utils.Sequence object and extract only the target (y)
    In this case, return a np.array of masks which can then be flattened
    """
    test_image_iterator = iter(test_sequence)

    masks = []
    for _, mask in test_image_iterator:
        masks.extend([*mask])

    masks = np.array(masks)
    return masks


def make_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def get_class_f1_score(y_true, y_pred, class_label) -> np.float64:
    """
    From a true, predicted, and class label get the F1 score for any particular class

    Returns float of the F1 score (binary)
    """

    return f1_score(y_true, y_pred, labels=[class_label], average=None)
