import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.in_channels = [image_channels, 32, 64]
        # Set number of filters in conv layers
        self.num_filters = [32, 64, 128]
        self.num_classes = num_classes
        # Define the convolutional layers
        modules = []
        for i in range(3):
            modules.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_filters[i],
                    kernel_size=5,
                    stride=1,
                    padding=2
                )
            )
            modules.append(
                nn.ReLU()
            )
            modules.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

        self.feature_extractor = nn.Sequential(*modules)

        # The output of feature_extractor will be [batch_size, 128, 4, 4]
        self.num_output_features = 128*4*4

        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

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
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")

    # 2b
    from trainer import compute_loss_and_accuracy

    dataloader_train, dataloader_val, dataloader_test = dataloaders

    _, train_accuracy = compute_loss_and_accuracy(
        dataloader_train, model, trainer.loss_criterion)
    _, val_accuracy = compute_loss_and_accuracy(
        dataloader_val, model, trainer.loss_criterion)
    _, test_accuracy = compute_loss_and_accuracy(
        dataloader_test, model, trainer.loss_criterion)

    print("Final training accuracy: ", train_accuracy)
    print("Final validation accuracy: ", val_accuracy)
    print("Final test accuracy: ", test_accuracy)
