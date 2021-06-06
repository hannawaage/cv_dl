import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


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

    # Divide every row by its maximum, i.e. normalize each row to the max.
    norm_out = outputs/(np.max(outputs, axis = 1)[:, None])

    # Turn of every element that isnt the one with the maximum likelihood
    norm_out[norm_out < 1] = 0
    predicted = norm_out

    # For correctly predictions, targets-predicted will give 0 in the place.
    # If wrongly predicted, we will get one 1, and one -1 in the same row.
    diff = targets - predicted

    # Rows with errors have two nonzero elements, hence divide by two.
    incorrect = np.count_nonzero(diff)/2
    N = X.shape[0]
    accuracy = (N - incorrect)/N

    return accuracy



class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        outputs = self.model.forward(X_batch)
        self.model.backward(X_batch, outputs, Y_batch)
        self.model.w = self.model.w - learning_rate*(self.model.grad)
        loss = cross_entropy_loss(Y_batch, outputs)

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)


    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)

    num_classes = 10

    fig, axs = plt.subplots(2, 10)
    fig.suptitle('Resulting weights with and without regularization')
    for i in range(num_classes):
        axs[0, i].imshow(model.w[:, i][1:].reshape(28, 28), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title(str(i))
        axs[1, i].imshow(model1.w[:, i][1:].reshape(28, 28), cmap='gray')
        axs[1, i].axis('off')

    plt.show()

    #plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]

    val_histories = []
    weight_norms = {}
    for i in l2_lambdas:
        # Intialize model
        model = SoftmaxModel(i)
        # Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        _, val_history = trainer.train(num_epochs)
        val_histories.append(val_history)
        weight_norms[i] = np.sum(np.square(model.w))

    plt.ylim([0.89, .93])
    for ind, val_hist in enumerate(val_histories):
        legend = "Validation accuracy with L2 Lambda value: " + str(l2_lambdas[ind])
        utils.plot_loss(val_hist["accuracy"], legend)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

   # Task 4d - Plotting of the l2 norm for each weight
    plt.scatter(range(len(weight_norms)), np.flip(list(weight_norms.values())))
    plt.xticks(range(len(weight_norms)), np.flip(list(weight_norms.keys())))
    plt.xlabel("Lambda")
    plt.ylabel("L2 norm")
    plt.legend()
    plt.show() 
