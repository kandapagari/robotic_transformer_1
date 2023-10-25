# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np
import torch

from robotic_transformer import RT1
from robotic_transformer.workflow.utils import resize_images


@dataclass
class Input:
    input_images: list[np.ndarray]
    sentences: list[str] | str


def infer(
    inputs: Input,
    device: str | None = 'cpu',
    num_actions: int = 7,
    backbone_name: str = 'efficientnet-b3',
) -> torch.Tensor:
    input_tensor = inputs.input_images
    sentences = inputs.sentences if isinstance(
        inputs.sentences, list,
    ) else [inputs.sentences]
    model = RT1(
        num_actions=num_actions,
        backbone_name=backbone_name,
    ).to(device)
    input_tensor = resize_images(input_tensor, device=device)
    return model(input_tensor, sentences)


if __name__ == '__main__':
    input_tensor = [
        np.zeros(shape=(224, 224, 3), dtype=np.uint8)
        for _ in range(6)
    ]
    sentences = ["Pick apple from top drawer and place on counter."]
    output_tensor = infer(Input(input_tensor, sentences))
    print(output_tensor.shape)
