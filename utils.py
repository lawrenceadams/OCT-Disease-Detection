from typing import Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
    test_image_iterator = iter(test_sequence)

    masks = []
    f1_macro = []
    f1_micro = []
    for _, mask in test_image_iterator:
        masks.extend([*mask])

    masks = np.array(masks)

    for mask, prediction in zip(masks, predictions):
        f1_macro.append(f1_score(mask.flatten(), prediction.flatten(), average="macro"))
        f1_micro.append(f1_score(mask.flatten(), prediction.flatten(), average="micro"))

    print(f"Macro F1: {np.average(f1_macro)}")
    print(f"Micro F1: {np.average(f1_micro)}")
