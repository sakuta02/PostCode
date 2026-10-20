from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from dataset import IMAGENET_MEAN, IMAGENET_STD, PadToSquare
from masks import BACKGROUND, IGNORE_INDEX, majority_pool, pad_to_square

FILL = (124, 116, 104)


def sample_crop(side: int, scale=(0.85, 1.0), ratio=(0.92, 1.09), tries: int = 10):
    area = side * side
    for _ in range(tries):
        target = area * random.uniform(*scale)
        log_r = random.uniform(math.log(ratio[0]), math.log(ratio[1]))
        ar = math.exp(log_r)
        w = int(round(math.sqrt(target * ar)))
        h = int(round(math.sqrt(target / ar)))
        if 0 < w <= side and 0 < h <= side:
            return random.randint(0, side - w), random.randint(0, side - h), w, h
    return 0, 0, side, side


class JointTransform:
    def __init__(self, img_size: int, grid: int, train: bool = True,
                 strength: float = 1.0, grid_mult: int = 4, hflip: bool = False):
        self.img_size = img_size
        self.grid = grid
        self.mask_size = grid * grid_mult
        self.train = train
        self.s = max(0.0, strength)
        self.force_hflip = hflip
        self.pad_img = PadToSquare(FILL)

    def geometry(self, img: Image.Image, mask: Image.Image | None):
        img = self.pad_img(img)
        if mask is not None:
            mask = pad_to_square(mask, BACKGROUND)
        side = img.size[0]

        if self.train:
            left, top, w, h = sample_crop(side, scale=(1.0 - 0.15 * self.s, 1.0))
            img = img.crop((left, top, left + w, top + h)).resize(
                (self.img_size, self.img_size), Image.BICUBIC)
            if mask is not None:
                k = mask.size[0] / side
                mask = mask.crop((round(left * k), round(top * k),
                                  round((left + w) * k), round((top + h) * k))).resize(
                    (self.mask_size, self.mask_size), Image.NEAREST)

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask is not None:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            if self.s > 0 and random.random() < 0.3 * self.s:
                angle = random.uniform(-7 * self.s, 7 * self.s)
                img = img.rotate(angle, Image.BILINEAR, fillcolor=FILL)
                if mask is not None:
                    mask = mask.rotate(angle, Image.NEAREST, fillcolor=IGNORE_INDEX)

            if self.s > 0:
                for enhancer, amount in ((ImageEnhance.Brightness, 0.30),
                                         (ImageEnhance.Contrast, 0.30),
                                         (ImageEnhance.Color, 0.20)):
                    f = 1.0 + random.uniform(-amount, amount) * self.s
                    img = enhancer(img).enhance(f)
                if random.random() < 0.20 * self.s:
                    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        else:
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
            if mask is not None:
                mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
            if self.force_hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask is not None:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if mask is None:
            grid = np.full((self.grid, self.grid), IGNORE_INDEX, dtype=np.uint8)
        else:
            grid = majority_pool(np.asarray(mask), self.grid)
        return img, grid

    def __call__(self, img: Image.Image, mask: Image.Image | None):
        import torch

        img, grid = self.geometry(img, mask)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return x, torch.from_numpy(grid.astype(np.int64))
