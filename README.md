# CXR diagnosis on the CheXpert Dataset

### Preparation

* Download the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/).
* Change the directories in "dataset.py",  "feature_extraction.py" accordingly.

### Training

Train a ConvNet on the CheXpert dataset:

> python train.py --config main_config.json --arch resnet18 --device 0 --pretrained_imagenet --strategy U-Ones --exp_name resnet18_ones_adam_1e4 --lr 0.0001 --optimizer Adam --batch_size 16

### Feature extraction:

Extract CNN features using a previously computed model checkpoint:

> python feature_extraction.py --device 2 --exp_name vgg11bn_ones_adam_1e4_64 --arch vgg11_bn --mode train --checkpoint /gpu-data2/jpik/CheXpert/checkpoints/vgg11bn_ones_adam_1e4_64/0218_102921/model_best.pth --strategy U-Ones
