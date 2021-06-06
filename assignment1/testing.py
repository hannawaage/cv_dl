
import numpy as np
import matplotlib.pyplot as plt
from task3a import SoftmaxModel

def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """

    outputs = model.forward(X)
    norm_out = outputs/(np.max(outputs, axis = 1)[:, None])
    norm_out[norm_out < 1] = 0

    predicted = norm_out
    diff = targets - predicted
    print(diff)
    incorrect = np.count_nonzero(diff)
    print(incorrect)
    N = X.shape[0]
    print(N)
    accuracy = (N - incorrect)/N

    return accuracy


targets = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
X = np.zeros((1, 785))
model = SoftmaxModel(0)

print(calculate_accuracy(X, targets, model))
