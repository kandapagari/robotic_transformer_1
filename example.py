# -*- coding: utf-8 -*-
import torch

from robotic_transformer import RT1
from robotic_transformer.data import TFRecordDataset
from robotic_transformer.workflow.utils import resize_images

if __name__ == "__main__":
    input_tensor = torch.randn(6, 3, 224, 224)
    sentences = ["Pick apple from top drawer and place on counter."]
    data_dir = "/data/net/ml_data/google_dataset/rt-1-data-release"
    split = 'train'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    tf_record_dataset = TFRecordDataset(data_dir, split)
    model = RT1(num_actions=7, backbone_name='efficientnet-b3').to(device)
    for item in iter(tf_record_dataset):
        images, actions, info, instruction = item
        input_tensor = resize_images(images[:6], device=device)
        sentences = [instruction]
        output_tensor = model(input_tensor, sentences)
        print(output_tensor.shape)
