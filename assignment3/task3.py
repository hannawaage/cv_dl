import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


def out_size(in_size, kernel_size, padding, stride):
  s = (in_size - kernel_size + 2*padding)/stride + 1
  return s


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 image_size,
                 filters_per_layer,
                 activation,
                 filter_size,
                 filter_stride,
                 filter_padding,
                 neurons_per_layer,
                 dropout=False,
                 batch_norm = False,
                 normal_init = False,
                 pool_size=0,
                 pool_stride=0,
                 ):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # Define parameteres for conv layers
        self.filters_per_layer = filters_per_layer
        self.in_channels = [image_channels] + filters_per_layer[:-1]
        self.activation = activation
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.filter_padding = filter_padding

        # Define parameters for affine layers
        self.num_classes = neurons_per_layer[-1]
        self.neurons_per_layer = neurons_per_layer

        # Parameters for pooling layers
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        in_size = image_size
        # Define the convolutional layers
        conv_modules = []
        for layer in range(len(filters_per_layer)):
            conv_layer = nn.Conv2d(
                in_channels=self.in_channels[layer],
                out_channels=self.filters_per_layer[layer],
                kernel_size=self.filter_size,
                stride=self.filter_stride,
                padding=self.filter_padding
            )
            if normal_init:
                torch.nn.init.kaiming_normal_(
                    conv_layer.weight,  nonlinearity='relu')
            conv_modules.append(
                conv_layer
            )
            if batch_norm:
                conv_modules.append(
                    nn.BatchNorm2d(self.filters_per_layer[layer])
                )
            size_out = out_size(in_size, self.filter_size,
                                self.filter_padding, self.filter_stride)
            conv_modules.append(
                activation
            )
            if pool_size:
              conv_modules.append(
                  nn.MaxPool2d(
                      kernel_size=self.pool_size,
                      stride=self.pool_stride
                  )
              )
              size_out = out_size(
                  size_out, self.pool_size, 0, self.pool_stride)
            if dropout:
                conv_modules.append(
                    nn.Dropout(0.25)
                )
            in_size = size_out

        self.feature_extractor = nn.Sequential(*conv_modules)
        self.num_output_features = int(self.filters_per_layer[-1]*size_out**2)

        # Define the affine layers
        classifier_modules = []
        current_in = self.num_output_features
        for layer in range(len(neurons_per_layer) - 1):
            classifier_modules.append(
                nn.Linear(current_in, neurons_per_layer[layer])
            )
            if batch_norm:
                classifier_modules.append(
                    nn.BatchNorm1d(neurons_per_layer[layer])
                )
            classifier_modules.append(
                activation
            )
            current_in = neurons_per_layer[layer]

        classifier_modules.append(nn.Linear(current_in, self.num_classes))

        self.classifier = nn.Sequential(*classifier_modules)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        feature_maps = self.feature_extractor(x)
        feature_maps = feature_maps.view(-1, self.num_output_features)
        out = self.classifier(feature_maps)

        batch_size = out.shape[0]
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(
        trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(
        trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    print("CUDA available: ", torch.cuda.is_available())
    utils.set_seed(0)

    # Model 1

    epochs = 10
    batch_size = 64
    learning_rate = 0.05
    early_stop_count = 4
    augment = False
    dataloaders = load_cifar10(batch_size, augment)

    # Base model 1
    filters = [64, 64, 64]
    relu = nn.ReLU() 
    neurons = [64, 10]
    """ Best model:
    filters = [128, 128, 128, 128]
    relu = nn.LeakyReLU(0.2)
    neurons = [64, 10] """

    model_1 = ExampleModel(
        image_channels=3,
        image_size=32,
        filters_per_layer=filters,
        activation=relu,
        filter_size=5,
        filter_stride=1,
        filter_padding=2,
        neurons_per_layer=neurons,
        pool_size=2,
        pool_stride=2,
    )
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_1,
        dataloaders
    )
    trainer.train()

    create_plots(trainer, "task3")

    # Print statistics
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_accuracy = compute_loss_and_accuracy(
        dataloader_train, model_1, trainer.loss_criterion)
    _, val_accuracy = compute_loss_and_accuracy(
        dataloader_val, model_1, trainer.loss_criterion)
    _, test_accuracy = compute_loss_and_accuracy(
        dataloader_test, model_1, trainer.loss_criterion)

    print("Model 1 baseline statistics: ")
    print("Final training loss: ", train_loss)
    print("Final training accuracy: ", train_accuracy)
    print("Final validation accuracy: ", val_accuracy)
    print("Final test accuracy: ", test_accuracy)

    # Model 2

    epochs = 10
    batch_size = 64
    learning_rate = 0.05
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)

    filters = [32, 64, 128]
    relu = nn.ReLU()
    neurons = [64, 10]

    model_2 = ExampleModel(
        image_channels=3,
        image_size=32,
        filters_per_layer=filters,
        activation=relu,
        filter_size=4,
        filter_stride=2,
        filter_padding=1,
        neurons_per_layer=neurons,
    )
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_2,
        dataloaders
    )
    trainer.train()

    create_plots(trainer, "task3")

    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_accuracy = compute_loss_and_accuracy(
        dataloader_train, model_2, trainer.loss_criterion)
    _, val_accuracy = compute_loss_and_accuracy(
        dataloader_val, model_2, trainer.loss_criterion)
    _, test_accuracy = compute_loss_and_accuracy(
        dataloader_test, model_2, trainer.loss_criterion)

    print("Model 2 baseline statistics: ")
    print("Final training loss: ", train_loss)
    print("Final training accuracy: ", train_accuracy)
    print("Final validation accuracy: ", val_accuracy)
    print("Final test accuracy: ", test_accuracy)
