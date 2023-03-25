# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import PIL
from PIL import ImageFilter
from torchvision.transforms import transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        isPIL = isinstance(x, PIL.Image.Image)
        if not isPIL:
            x = transforms.ToPILImage()(x)

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        if not isPIL:
            x = transforms.ToTensor()(x)

        return x


class ScaleTransform(object):
    """Scales the pixel values to between 0 and 1"""

    def __call__(self, image):
        return image / 255
