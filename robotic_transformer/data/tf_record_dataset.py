# -*- coding: utf-8 -*-
import tensorflow_datasets as tfds
from torch.utils.data import Dataset


class TFRecordDataset(Dataset):
    """A PyTorch dataset class for loading data from TensorFlow Records
    (TFRecords) stored in a directory.

    Args:
        data_dir (str): The directory containing the TFRecords.
        split (str): The split of the dataset, e.g., 'train' or 'test'.

    Attributes:
        data_dir (str): The directory containing the TFRecords.
        split (str): The split of the dataset, e.g., 'train' or 'test'.
        builder (tfds.core.DatasetBuilder): A TensorFlow Datasets (TFDS) dataset builder.
        loaded_dataset (tf.data.Dataset): The loaded dataset from TFDS.

    Methods:
        __len__(self): Returns the number of examples in the dataset.
        __getitem__(self, idx): Retrieves a single example (images, actions, info, instruction)
        at the given index.

    Note:
        This class is designed to load data from TFRecords using TensorFlow Datasets (TFDS) and
        convert it into a PyTorch-compatible dataset. It is assumed that the TFRecords in the
        specified directory match the structure and format expected by the TFDS builder.
    """

    def __init__(self, data_dir: str, split: str = 'train') -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.builder = tfds.builder_from_directory(builder_dir=data_dir)
        self.builder.download_and_prepare()
        self.loaded_dataset = self.builder.as_data_source()

    def __len__(self):
        return self.loaded_dataset[self.split].length

    def __getitem__(self, idx) -> tuple[list, list, list, str]:
        """Get an item (sample) from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple of (images, actions, info, instruction).

        Notes:
            This method retrieves an item (sample) from the dataset by index. It returns a tuple
            containing images, actions, info, and the natural language instruction for the sample.
        """
        images = []
        actions = []
        info = []
        steps = self.loaded_dataset[self.split][idx]['steps']
        instruction = list(steps)[0]['observation']['natural_language_instruction'].decode(
            'utf-8',
        )
        for step in steps:
            images.append(step['observation']['image'])
            actions.append(step['action'])
            info.append(step['info'])
        assert len(images) == len(actions) == len(
            info,
        ), "Error loading dataset"
        return images, actions, info, instruction


if __name__ == '__main__':
    dataset_name = "fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor"  # NOQA
    data_dir = "/data/net/ml_data/google_dataset/robotics/bc_z/1.0.1/"
    split = 'train'
    tf_record_dataset = TFRecordDataset(data_dir, split)
    for item in iter(tf_record_dataset):
        images, actions, info, instruction = item
        print(len(images))
    # images, actions, info, instruction = next(iter(tf_record_dataset))
    # images_1, actions_1, info_1, instruction_1 = next(iter(tf_record_dataset))
    print()
