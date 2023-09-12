# Flower Classification with PyTorch

This repository contains code to train a deep learning model for flower classification and predict flower classes using a trained model.

## Requirements

- PyTorch
- torchvision
- PIL
- numpy
- tqdm
- matplotlib
- argparse
- json

## Dependencies Installation

### Using Conda

```
conda env create -f environment.yml
conda activate torch_env
```

### Using pip

```
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

To train a new model using your dataset:

```bash
python train.py <data_directory> [OPTIONS]
```

### Arguments

- `data_directory`: Path to data directory (contains 'train' & 'valid').
- `--save_dir`: Save checkpoints here (default: `.`).
- `--arch`: Architecture [vgg16] (default: `vgg16`).
- `--learning_rate`: Learning rate (default: `0.001`).
- `--hidden_units`: Hidden units (default: `512`).
- `--epochs`: Epochs (default: `1`).
- `--gpu`: Use GPU.
- `--mps`: Use MPS on Mac.

### Invocation Example

```
(torch_env) kyle@Kyles-MacBook-Pro aipnd-project % python3 train.py flowers --mps

/Users/kyle/anaconda3/envs/torch_env/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/kyle/anaconda3/envs/torch_env/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Epoch 1/1 [Train]:  48%|███████████████████████▎                         | 49/103 [00:13<00:09,  5.75it/s]Epoch 1/1.. Train loss: 4.659
Epoch 1/1 [Train]:  96%|███████████████████████████████████████████████  | 99/103 [00:21<00:00,  5.74it/s]Epoch 1/1.. Train loss: 3.761
Epoch 1/1.. Validation loss: 2.821.. Validation accuracy: 0.414
/Users/kyle/anaconda3/envs/torch_env/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:149: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
Training finished!

```

which flowers data folder can be retrieved from

```

curl -O 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
mkdir flowers && tar -xzf flower_data.tar.gz -C flowers

```

## Prediction

To predict the flower class of an image:

```

python predict.py <image_path> <checkpoint> [OPTIONS]

```

### Invocation Example

```

(torch_env) kyle@Kyles-MacBook-Pro aipnd-project % python3 predict.py image.jpg checkpoint.pth --mps --category_names cat_to_name.json
/Users/kyle/anaconda3/envs/torch_env/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/kyle/anaconda3/envs/torch_env/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Probabilities: [0.05254214257001877, 0.04158187285065651, 0.03718974068760872, 0.030157823115587234, 0.028122209012508392]
Classes: ['rose', 'lotus lotus', 'bougainvillea', 'petunia', 'pelargonium']

```

### Arguments

- `image_path`: Image path.
- `checkpoint`: Model checkpoint path.
- `--top_k`: Top K classes (default: `5`).
- `--category_names`: JSON categories path.
- `--gpu`: Use GPU.
- `--mps`: Use MPS on Mac.

## Model

Default is `vgg16` with custom classifier:

1. Linear: 25088 -> `hidden_units`
2. ReLU
3. Dropout (0.5)
4. Linear: `hidden_units` -> 102
5. LogSoftmax

## Optimizations

- **Gradient Accumulation**: For larger batch sizes.
- **Learning Rate Scheduler**: Reduces rate if validation loss plateaus.

## Early Stopping

Stops if no validation loss improvement after `patience` epochs (default: `5`).

## Author

- **Kyle Law**
- **Udacity**

## License

MIT
