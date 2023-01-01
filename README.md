# CXR diagnosis on the CheXpert Dataset

### Preparation

* Download the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/).
* Change the directories in "dataset.py",  "feature_extraction.py" accordingly.

### Training

Train a ConvNet on the CheXpert dataset:

```
python train.py --config main_config.json --arch resnet18 --device 0 --pretrained_imagenet --strategy U-Ones --exp_name resnet18_ones_adam_1e4 --lr 0.0001 --optimizer Adam --batch_size 16
```

### Feature extraction:

Extract CNN features using a previously computed model checkpoint:

```
python feature_extraction.py --device 2 --exp_name vgg11bn_ones_adam_1e4_64 --arch vgg11_bn --mode train --checkpoint [CHECKPOINT_PATH] --strategy U-Ones
```

### Citation

If you use CheXpert for your research, consider citing the original paper:

```
@inproceedings{irvin2019chexpert,
  title={{CheXpert}: A large chest radiograph dataset with uncertainty labels and expert comparison},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and others},
  booktitle={Proc. AAAI Conf. on Artificial Intelligence},
  volume={},
  number={},
  pages={},
  year={2019}
}
```

### Acknowlegements

* [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)

### Contact

For questions feel free to open an issue.
