# Robotic Transformer with PyTorch

<!-- <video width="320" height="240" controls>
  <source src="[video.mov](https://robotics-transformer1.github.io/img/RT1-video.mp4)" type="video/mp4">
</video> -->

<img src="https://pytorch.org/assets/images/pytorch-logo.png", width="200"/>

## Overview

This Python project implements a Robotic Transformer model using PyTorch. The Robotic Transformer is a deep learning architecture designed for robot perception and control tasks. It leverages the power of self-attention mechanisms to process sequences of sensor data efficiently.

## Table of Contents

- [Robotic Transformer with PyTorch](#robotic-transformer-with-pytorch)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Training](#training)
    - [TODO](#todo)
  - [Inference](#inference)
    - [TODO](#todo-1)
  - [Contributing](#contributing)
  - [License](#license)

## Prerequisites

Before using this project, you should have the following dependencies installed:

- Python 3.11+
- Conda
- Poetry

You can install these packages using pip:

```bash
conda create -n rt1-torch python=3.11 -y
conda activate rt1-torch
pip install poetry -U

```

## Installation

1. Clone the repository:

   ```bash
   git clone git@git.ar.int:dev/ai/ml/agilebrain/lrm/rt1-torch.git
   ```

1. Change directory to the project folder:

   ```bash
   cd rt1-torch
   ```

1. Install the project dependencies:

   ```bash
   poetry install
   ```

## Usage

To use the Robotic Transformer model in your own Python code, you can import it as follows:

```python
from robotic_transformer import RT1

# Create an instance of the RoboticTransformer model
model = RT1(num_actions=7, backbone_name='efficientnet-b3').to(device)

# Use the model for your robotic perception or control task
```

You can adjust the hyperparameters (`num_actions`, `backbone_name` etc.) to suit your specific application.

## Training

### TODO

This project also includes sample code for training the Robotic Transformer model. You can train the model using your own dataset by modifying the training script and data loaders. To start training, run:

```bash
python robotic_transformer/workflow/train.py
```

Make sure to customize the data loading and training settings in the `train.py` script for your specific use case.

## Inference

### TODO

Once you have a trained model, you can use it for inference on new data. Modify the inference script to load your trained model weights and perform inference on your robot's sensor data.

To run inference:

```bash
python robotic_transformer/workflow/inference.py
```

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
1. Create a new branch for your feature or bug fix: `git checkout -b feature/my-new-feature`.
1. Make your changes and commit them: `git commit -m 'Add new feature'`.
1. Push to the branch: `git push origin feature/my-new-feature`.
1. Create a pull request on the original repository.

Please ensure your code follows the project's coding style and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy Robotics with Robotic Transformer! If you have any questions or encounter any issues, feel free to open an issue on the GitHub repository.
