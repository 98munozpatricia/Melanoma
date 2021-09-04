# Melanoma detection CNN using PyTorch

Follow these steps to train and optimiza a Melanoma Dataset with EfficientNet b0.

This project solves the problem proposed by SIIM-ISIC: [Melanoma Classification in Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification).

## Installation

Clone this repository
````bash
git clone https://github.com/98munozpatricia/Melanoma.git
cd Melanoma/
````
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Efficientnet and Torch toolbox.

```bash
pip install efficientnet_pytorch
pip install torchtoolbox
pip install torch_pruning
```
Download the dataset from [Google Drive](https://drive.google.com/file/d/1Nwsx8mEotwKCImB1cTJ7fqaOXjLNvmoW/view?usp=sharing) and unzip them in the Melanoma folder using the following command:
```bash
unzip dataset.zip
```

## Usage

To train the model simply execute the training script:
```python
python trainMelanoma.py
```
This aim of this project is also to optimize the model. For doing so, two other scripts are added: quantization.py and pruning.py.
Those are developed for the Melanoma model optimization, but can be used as an example for any other Neural Network model by changing the application layers.
Every script has different methods for each type of quantization including:

* Dynamic quantization
* Static quantization
* Quantization-aware training

* Unstructured pruning
* Structured pruning

These methods use the PyTorch API for [Quantization](https://pytorch.org/docs/stable/quantization.html) and [Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html).

In the pruning example, it also has a method using [Torch-Pruning Github](https://github.com/VainF/Torch-Pruning) to overcome PyTorch dense tensors limitations.
