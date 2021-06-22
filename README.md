# Training Melanoma Dataset

Follow these steps to train Melanoma Dataset with EfficientNet b0

## Installation

Clone this repository
````bash
git clone https://github.com/98munozpatricia/Melanoma.git
cd Melanoma/
````
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Efficientnet and Torch toolbox.

```bash
pip3 install efficientnet_pytorch
pip3 install torchtoolbox
```
Download the dataset from [Google Drive](https://drive.google.com/file/d/1Nwsx8mEotwKCImB1cTJ7fqaOXjLNvmoW/view?usp=sharing) and unzip them in the Melanoma folder using the following command:
```bash
unzip dataset.zip
```

## Usage

```python
python trainMelanoma.py
```
