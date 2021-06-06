import numpy as np
import utils
import typing
np.random.seed(1)

def get_training_statistics(train: np.ndarray):
    mean = np.mean(train)
    std = np.std(train)
    return np.array([mean, std])

def pre_process_images(X: np.ndarray, statistics: np.array):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    # Normalize
    mean = statistics[0]
    std = statistics[1]

    X = X - mean
    X = X/(std)

    # Add bias
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)

    return X

def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    cross_entropy = -np.sum(targets*np.log(outputs), axis=1)
    return cross_entropy.mean()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_improved(z):
    return 1.7159*np.tanh(2*z/3)

def dsigmoid(a):
    return a*(1 - a)

def dsigmoid_improved(a):
    return 1.7159*(2/3)*(1-(a/(1.7159))**2)


def softmax(z):
    exp_mat = np.exp(z)
    # Divide every row by the sum of all elements in the row
    return exp_mat/np.sum(exp_mat, axis=1, keepdims=True)

class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        self.hidden_layer_output = []

        # Initialize the weights
        self.ws = []
        self.delta_ws = []
        prev = self.I
        mu = 0
        for size in self.neurons_per_layer:   
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if self.use_improved_weight_init:
                std = 1/np.sqrt(prev)
                w = np.random.normal(mu, std, size=w_shape)
            else:
                w = np.random.uniform(-1, 1, size=w_shape)
            self.ws.append(w)
            delta_w = np.zeros(w_shape)
            self.delta_ws.append(delta_w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        a_h = X
        self.hidden_layer_output = [X]
        n_hidden_layers = len(self.neurons_per_layer) - 1
        for layer in range(n_hidden_layers):
            z_h = a_h @ self.ws[layer]
            if self.use_improved_sigmoid:
                a_h = sigmoid_improved(z_h)
            else:
                a_h = sigmoid(z_h)
            self.hidden_layer_output.append(a_h)
        
        z_k = a_h @ self.ws[-1]
        output = softmax(z_k)

        return output

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        norm = 1/targets.shape[0]
        
        a_h = self.hidden_layer_output
        delta_next = -(targets - outputs)
        self.grads[-1] = norm*(a_h[-1].T @ delta_next)
        n_layers = len(self.neurons_per_layer)
        for layer in range(n_layers - 1):
            L = n_layers - 2 - layer
            dot_prod = delta_next @ self.ws[L + 1].T
            a_current = a_h[L + 1]
            if self.use_improved_sigmoid:
                ds = dsigmoid_improved(a_current)
            else:
                ds = dsigmoid(a_current)
            delta_current = dot_prod*ds
            a_prev = a_h[L]
            self.grads[L] = norm*(
                a_prev.T @ delta_current)
            delta_next = delta_current

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    OH = np.ndarray((Y.shape[0], num_classes))
    for ind, i in enumerate(Y):
        OH[ind] = [1 if i == j else 0 for j in range(num_classes)]

    return OH


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    statistics = get_training_statistics(X_train)
    X_train = pre_process_images(X_train, statistics)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)
    gradient_approximation_test(model, X_train, Y_train)
