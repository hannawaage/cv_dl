import pathlib
import matplotlib.pyplot as plt
import utils
import torch
import torchvision
from torch import nn
from dataloaders4 import load_cifar10
from restnet_trainer import ResNet18_Trainer, compute_loss_and_accuracy

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)

        for param in self.model.parameters():
            param.required_grad = False
        for param in self.model.fc.parameters():
            param.required_grad = True
        for param in self.model.layer4.parameters():
            param.required_grad = True

    def forward(self, x):
        x = self.model(x)
        return x 

def create_plots(trainer: ResNet18_Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def print_accuracy(dataloader: torch.utils.data.DataLoader):
    dataloader_train, dataloader_val, dataloader_test = dataloader

    _, training_accuracy = compute_loss_and_accuracy(dataloader_train, model, trainer.loss_criterion)
    _, validation_accuracy = compute_loss_and_accuracy(dataloader_val, model, trainer.loss_criterion)
    _, test_accuracy = compute_loss_and_accuracy(dataloader_test, model, trainer.loss_criterion)

    print("Final training accuracy:", training_accuracy)
    print("Final validation accuracy:", validation_accuracy)
    print("Final test accuracy:", test_accuracy)


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders4 = load_cifar10(batch_size)
    model = Model()
    trainer = ResNet18_Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders4
    )
    trainer.train()
    create_plots(trainer, "task4")
    print_accuracy(dataloaders4)




