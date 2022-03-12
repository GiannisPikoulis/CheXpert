import sys
import argparse
import collections
import torch
import torchvision
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer.trainer import Trainer
from dataset import CheXpertDataset
from model.models import Model

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args, config):
    
    logger = config.get_logger('train') 
    
    model = Model(logger=logger, num_classes=args.num_classes,
                  arch=args.arch, partial_bn=args.partial_bn,
                  pretrained_imagenet=args.pretrained_imagenet)
    
    logger.info("\nTotal number of network trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logger.info(model)    
            
    mean = model.mean
    std = model.std   
    policies = model.get_optim_policies()

    train_dataset = CheXpertDataset(mode="train", version='small', label_strategy=args.strategy,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((256, 256)),
                                    torchvision.transforms.RandomCrop((224, 224)),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean, std)
                                    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

    val_dataset = CheXpertDataset(mode="valid", version='small', label_strategy=args.strategy,
                                  transform=torchvision.transforms.Compose([
                                  torchvision.transforms.Resize((224, 224)),
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(mean, std)
                                  ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(policies, lr=args.lr, weight_decay=args.weight_decay)

    """
    Starting epoch is set to 1
    Consider the fact that the learning rate is reduced one epoch after each milestone
    """
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

	# get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss_categorical'])
    metrics = [getattr(module_metric, met) for met in config['metrics_categorical']]

    trainer = Trainer(model, criterion, metrics, optimizer, config=config, train_dataloader=train_loader, val_dataloader=val_loader, lr_scheduler=lr_scheduler)

    trainer.train()    
    logger.info('Best result: {}'.format(trainer.mnt_best))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CNN training on the CheXpert dataset')
	
	# ========================= Runtime Configs ==========================
    parser.add_argument('--n_workers', default=4, type=int, help='number of data loading workers (default: %(default)s)')
    parser.add_argument('--config', default=None, type=str, help='config file path (default: %(default)s)')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: %(default)s)')
    parser.add_argument('--device', required=True, type=str, help='indices of GPUs to enable separated by commas')

	# ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="resnet18", choices=["mlp", "resnet18", "resnet34", "resnet50", "resnet101", "alexnet", "vgg11_bn", "densenet121"], help="CNN backbone architecture (default: %(default)s)")
    parser.add_argument('--strategy', type=str, default="U-Zeros", choices=["U-Zeros", "U-Ones"], help="Uncertain condition label replacement strategy (default: %(default)s)")

	# ========================= Learning Configs ==========================
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: %(default)s)')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate (default: %(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay (default: %(default)s)')
    parser.add_argument('--partial_bn', default=False, action="store_true", help='partial batch normalization (default: %(default)s)')
    parser.add_argument('--num_classes', type=int, default=14, help='number of emotional classes (default: %(default)s)')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer used during learning (default: %(default)s)')
    
	# ========================= Pretraining Configs ==========================
    parser.add_argument('--pretrained_imagenet', default=False, action="store_true", help='load ImageNet pretrained weights, for RGB body stream and all Flow/RGBDiff streams (default: %(default)s)')
    
	# custom cli options to modify configuration from default values given in json file.
    custom_name = collections.namedtuple('custom_name', 'flags type target help')
    custom_epochs = collections.namedtuple('custom_epochs', 'flags type target help')
    custom_milestones = collections.namedtuple('custom_milestones', 'flags type nargs target help')
    
    options = [custom_name(['--exp_name'], type=str, target='name', help="custom experiment name (overwrites 'name' value from the configuration file"), 
               custom_epochs(['--epochs'], type=int, target='trainer;epochs', help="custom number of epochs (overwrites 'trainer->epochs' value from the configuration file"), 
               custom_milestones(['--milestones'], type=int, nargs='+', target='lr_scheduler;args;milestones', help="custom milestones for scheduler (overwrites 'lr_scheduler->args->milestones' value from the configuration file")]
    
    config = ConfigParser.from_args(parser, options)
 
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
                   
    main(args, config)
