# -*- coding: utf-8 -*-
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import ToTensor


def resize_images(
    images, size=(224, 224),
    device='cpu',
    sampling: Resampling = Image.Resampling.NEAREST,
) -> torch.TensorType:
    resized_images = []
    for image in images:
        _image = Image.fromarray(image)
        _image = _image.resize(size, sampling)
        resized_images.append(ToTensor()(_image))
    resized_tensor: torch.TensorType = torch.as_tensor(
        np.array(resized_images),
    )
    return resized_tensor.to(device)
