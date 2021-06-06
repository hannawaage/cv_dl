import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, get_training_statistics
from task2 import SoftmaxTrainer


def plot_comp(train_history_with, train_history_without, val_history_with, val_history_without, legend):
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0, .4])

    utils.plot_loss(train_history_with["loss"],
                    "Train loss w/ " + legend, npoints_to_average=10)
    utils.plot_loss(
        train_history_without["loss"], "Train loss without " + legend, npoints_to_average=10)
    plt.ylabel("Training loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .97])
    utils.plot_loss(val_history_with["accuracy"],
                    "Validation accuracy w/ " + legend)
    utils.plot_loss(
        val_history_without["accuracy"], "Validation accuracy without " + legend)
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    statistics = get_training_statistics(X_train)
    X_train = pre_process_images(X_train, statistics)
    X_val = pre_process_images(X_val, statistics)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Make 3 comparisons:

    # With and without: 
    # 1. weight init
    # 2. improved sigmoid
    # 3. momentum 

    use_improved_weight_init = False
    use_improved_sigmoid = False
    use_momentum = False

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    A = np.array([[1, 0 , 0], [1, 1, 0], [1, 1, 1]])
    legends = ["weight init",
               "improved sigmoid",
               "momentum"]

    for i in range(3):
        if i == 2:
            learning_rate = .02

        use_improved_weight_init = A[i, 0]
        use_improved_sigmoid = A[i, 1]
        use_momentum = A[i, 2]

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        updated_train_history, updated_val_history = trainer.train(
            num_epochs)


        plot_comp(updated_train_history, train_history,
                  updated_val_history, val_history, legends[i], i)
        train_history = updated_train_history
        val_history = updated_val_history

