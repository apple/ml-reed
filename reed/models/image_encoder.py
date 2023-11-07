#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import typing as t

from abc import abstractmethod

import torch
from torch import nn

"""
Image encoders 
"""


def weight_init(m):
    """
    Initialize the weights of the image encoder following:
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class ImageEncoder(nn.Module):
    """
    Base image encoder for the reward models.
    """

    def __init__(self, obs_dim: t.List[int], out_size: int = 1,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 image_hidden_num_channels: int = 32,
                 *kwargs):
        """
        Encodes an image

        Args:
            obs_dim: dimensionality of the state images (height, width, channels)
            out_size: the size of the output
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
        """
        super(ImageEncoder, self).__init__()

        # track the dimensionality of the input, the output, and the hidden dimensions
        self._state_size = obs_dim
        self._out_size = out_size
        self._hidden_size = hidden_dim
        self._num_layers = hidden_depth
        self.image_hidden_num_channels = image_hidden_num_channels

        self.encoder = self._build()

        self.float()

    @abstractmethod
    def _build(self) -> nn.Module:
        pass

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass on the image

        Args:
            image: image or batch of images to be encoded

        Returns:
            embedding of the image(s)
        """
        return self.encoder(image)


class PixL2RImageEncoder(ImageEncoder):
    """
    from PixL2R: https://arxiv.org/pdf/2007.15543.pdf & https://github.com/prasoongoyal/PixL2R/blob/b0691be6b27e705a62534b58f97ff7b8b6655c7d/src/supervised/model.py#L52
    """
    def __init__(self, obs_dim: t.List[int], out_size: int = 1,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 image_hidden_num_channels: int = 32,
                 *kwargs):
        """
        Encodes an image

        Args:
            obs_dim: dimensionality of the state images (height, width, channels)
            out_size: the size of the output
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
        """
        super(PixL2RImageEncoder, self).__init__(
            obs_dim=obs_dim,
            out_size=out_size,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            final_activation=final_activation,
            image_hidden_num_channels=image_hidden_num_channels
        )

    def _build(self):
        return nn.Sequential(
            nn.Conv2d(self._state_size[0], self.image_hidden_num_channels, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.image_hidden_num_channels, self.image_hidden_num_channels, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.image_hidden_num_channels, self.image_hidden_num_channels, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))


class DRQv2ImageEncoder(ImageEncoder):
    """
    from drqv2: https://github.com/facebookresearch/drqv2/blob/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef/drqv2.py#L55
    """
    def __init__(self, obs_dim: t.List[int], out_size: int = 1,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 image_hidden_num_channels: int = 32,
                 *kwargs):
        """
        Encodes an image

        Args:
            obs_dim: dimensionality of the state images (height, width, channels)
            out_size: the size of the output
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
        """
        super(DRQv2ImageEncoder, self).__init__(
            obs_dim=obs_dim,
            out_size=out_size,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            final_activation=final_activation,
            image_hidden_num_channels=image_hidden_num_channels
        )

    def _build(self):
        return nn.Sequential(
                nn.Conv2d(self._state_size[0], self.image_hidden_num_channels, 3, stride=2),
                nn.ReLU(), nn.Conv2d(self.image_hidden_num_channels, self.image_hidden_num_channels, 3, stride=1),
                nn.ReLU(), nn.Conv2d(self.image_hidden_num_channels, self.image_hidden_num_channels, 3, stride=1),
                nn.ReLU(), nn.Conv2d(self.image_hidden_num_channels, self.image_hidden_num_channels, 3, stride=1),
                nn.ReLU())


def get_image_encoder(architecture: str, obs_dim: t.List[int], out_size: int = 1,
                      hidden_dim: int = 128, hidden_depth: int = 3,
                      image_hidden_num_channels: int = 32,
                      *kwargs) -> nn.Module:
    """
    Return the specified architecture initialized

    Args:
        architecture: which image encoder architecture to use
        obs_dim: dimensionality of the state images (height, width, channels)
        out_size: the size of the output
        hidden_dim: the size of the hidden layer(s)
        hidden_depth: the number of hidden layers
        image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
    Returns:
        initialized image encoder
    """
    if architecture == "pixl2r":
        # from PixL2R: https://arxiv.org/pdf/2007.15543.pdf & https://github.com/prasoongoyal/PixL2R/blob/b0691be6b27e705a62534b58f97ff7b8b6655c7d/src/supervised/model.py#L52
        return PixL2RImageEncoder(obs_dim=obs_dim, out_size=out_size,
                                  hidden_dim=hidden_dim, hidden_depth=hidden_depth,
                                  image_hidden_num_channels=image_hidden_num_channels)
    elif architecture == "drqv2":
        # from drqv2: https://github.com/facebookresearch/drqv2/blob/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef/drqv2.py#L55
        return DRQv2ImageEncoder(obs_dim=obs_dim, out_size=out_size,
                                 hidden_dim=hidden_dim, hidden_depth=hidden_depth,
                                 image_hidden_num_channels=image_hidden_num_channels)
    else:
        raise NotImplementedError(f"{architecture} is not an implemented image "
                                  f"encoder architecture")
