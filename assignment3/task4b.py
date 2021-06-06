
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from torch import nn
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]


#Task 4b)

#plot filter and activation of filter
plt.figure(figsize=(20, 8))
for idx, ind in enumerate(indices):
    plt.subplot(2, 5, idx + 1)
    image_filter = torch_image_to_numpy(first_conv_layer.weight[ind, ])
    plt.imshow(image_filter)
    plt.subplot(2, 5, idx + 6)
    image_activation = torch_image_to_numpy(activation[0, ind, :, :])
    plt.imshow(image_activation, cmap="Greys") 

plt.show()

#Task 4c)

ten_first_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Remove last layer, meaning the two blocks in the layer
last_conv_layer = nn.Sequential(*list(model.children())[:-2])
print("Last conv layer:", last_conv_layer)

#Forward through the three first layers
activation = last_conv_layer.forward(image)
#want shape 1x512x7x7
print("Activation shape:", activation.shape)

#plot activation of the filter
plt.figure(figsize=(20, 8))
for idx, ind in enumerate(ten_first_indices):
    plt.subplot(1, 10, idx + 1)
    image_activation = torch_image_to_numpy(activation[0, ind, :, :])
    plt.imshow(image_activation, cmap="Greys") 

plt.show()