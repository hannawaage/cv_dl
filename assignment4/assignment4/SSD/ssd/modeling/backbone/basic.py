import torch
from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        filters_layer1 = [64, 128, 128]
        activation = nn.LeakyReLU(0.2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=filters_layer1[0],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(filters_layer1[0]),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            activation,
            nn.Conv2d(
                in_channels=filters_layer1[0],
                out_channels=filters_layer1[1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(filters_layer1[1]),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            activation,

            ### Adding layers here (for getting the model to 90% mAP)

            nn.Conv2d(
                in_channels=filters_layer1[1],
                out_channels=filters_layer1[1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(filters_layer1[1]),
            activation,

            nn.Conv2d(
                in_channels=filters_layer1[1],
                out_channels=filters_layer1[1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(filters_layer1[1]),
            activation,

            nn.Conv2d(
                in_channels=filters_layer1[1],
                out_channels=filters_layer1[1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(filters_layer1[1]),
            activation,

            ###

            nn.Conv2d(
                in_channels=filters_layer1[1],
                out_channels=filters_layer1[1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(filters_layer1[1]),
            activation,
            nn.Conv2d(
                in_channels=filters_layer1[1],
                out_channels=output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(output_channels[0]),

        )

        conv_modules = []
        filters = [256, 512, 256, 256, 256]
        for layer in range(len(filters)):
            conv_modules.append([])
            conv_modules[layer].append(
                activation
            )
            conv_modules[layer].append(
                nn.Conv2d(
                    in_channels=output_channels[layer],
                    out_channels=filters[layer],
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            conv_modules[layer].append(
                nn.BatchNorm2d(filters[layer])
            )
            conv_modules[layer].append(
                activation
            )
            stride_val = 2
            pad_val = 1
            if layer == (len(filters) - 1):
                stride_val = 1
                pad_val = 0
            conv_modules[layer].append(
                nn.Conv2d(
                    in_channels=filters[layer],
                    out_channels=output_channels[layer + 1],
                    kernel_size=3,
                    stride=stride_val,
                    padding=pad_val
                )
            )
            conv_modules[layer].append(
                nn.BatchNorm2d(output_channels[layer + 1])
            )

        self.layer2 = nn.Sequential(*conv_modules[0])
        self.layer3 = nn.Sequential(*conv_modules[1])
        self.layer4 = nn.Sequential(*conv_modules[2])
        self.layer5 = nn.Sequential(*conv_modules[3])
        self.layer6 = nn.Sequential(*conv_modules[4])

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_features.append(self.layer1(x))
        out_features.append(self.layer2(out_features[0]))
        out_features.append(self.layer3(out_features[1]))
        out_features.append(self.layer4(out_features[2]))
        out_features.append(self.layer5(out_features[3]))
        out_features.append(self.layer6(out_features[4]))
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
