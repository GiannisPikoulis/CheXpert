import argparse
import time
import os
import sys
import numpy as np
import torchvision
import torch
import model.metric as module_metric
from dataset import CheXpertDataset
from model.models import Model
from utils import MetricTracker
from collections import OrderedDict

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def _prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids    

# options
parser = argparse.ArgumentParser(description="Feature extraction on CheXpert with CNNs")

parser.add_argument('--num_classes', type=int, default=14, help='number of emotional classes (default: %(default)s)')
parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "alexnet", "vgg11_bn", "densenet121"], help="CNN backbone architecture (default: %(default)s)")
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: %(default)s)')
parser.add_argument('--partial_bn', default=False, action="store_true", help='partial batch normalization (default: %(default)s)')
parser.add_argument('--strategy', type=str, required=True, choices=["U-Zeros", "U-Ones"], help="Uncertain condition label replacement strategy (default: %(default)s)")

parser.add_argument('--checkpoint', required=True, type=str, help='pretrained model checkpoint')
parser.add_argument('--output_dir', default='/gpu-data2/jpik/CheXpert/features', type=str, help='directory where to store outputs')
parser.add_argument('--exp_name', type=str, required=True, help='custom experiment name')

parser.add_argument('--n_workers', default=4, type=int, help='number of data loading workers (default: %(default)s)')
parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable separated by commas (default: all)')
parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use (default: %(default)s)')
#parser.add_argument('--save_outputs', default=False, action="store_true", help='whether to save outputs produced during inference (default: %(default)s)')
parser.add_argument('--mode', required=True, type=str, choices=['valid', 'train'], help='type of inference to run (default: %(default)s)')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
os.environ["CUDA_VISIBLE_DEVICES"] = args.device    

model = Model(logger=None, num_classes=args.num_classes,
              arch=args.arch, partial_bn=args.partial_bn,
              pretrained_imagenet=False, extract_features=True)

_outputs = []
_targets = []
_feats = []
    
mean = model.mean
std = model.std  

dataset = CheXpertDataset(mode=args.mode, version='small', label_strategy=args.strategy,
                          transform=torchvision.transforms.Compose([
                          torchvision.transforms.Resize((224, 224)),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean, std)
                          ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

print('\nSet: {}'.format(args.mode))

conditions = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", 
              "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", 
              "Fracture", "Support Devices"]
task_inds = [2, 5, 6, 8, 10]              

metrics = MetricTracker('mAP', 'mRA', "mRA (task)", "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", writer=None)

# Create directory to save predictions
if not os.path.exists(os.path.join(args.output_dir, args.exp_name, args.mode)):
    os.makedirs(os.path.join(args.output_dir, args.exp_name, args.mode)) 

# Load checkpoint    
print('Checkpoint path: {}'.format(args.checkpoint))     
checkpoint = torch.load(args.checkpoint)
                
new_state_dict = OrderedDict()

for k, v in checkpoint['state_dict'].items():
    if k[:7] == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
            
model.load_state_dict(new_state_dict)

# setup GPU device if available, move model into configured device
device, device_ids = _prepare_device(n_gpu_use=args.n_gpu)
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
        
print("Total number of network trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
model.eval()
            
with torch.set_grad_enabled(False):
    
    for batch_idx, batch_data in enumerate(dataloader):
        
        X = batch_data[0].to(device) 
        Y = batch_data[1].to(device)              
            
        out, feats = model(X)
            
        out = out.cpu().detach().numpy()
        _outputs.append(out)            
        targ = Y.cpu().detach().numpy()
        _targets.append(targ) 
        feats = feats.cpu().detach().numpy()
        _feats.append(feats)               

out = np.vstack(_outputs)    
target = np.vstack(_targets)
feats = np.vstack(_feats)

conds = conditions

if args.mode == 'valid':
    out = out[:, [x for x in range(14) if x != 12]]
    target = target[:, [x for x in range(14) if x != 12]]
    conds = conditions[:12] + [conditions[13]]

_ap = module_metric.average_precision(out, target)
_ra = module_metric.roc_auc(out, target)
metrics.update("mAP", np.mean(_ap))
metrics.update("mRA", np.mean(_ra))
for j in range(len(conds)):
    metrics.update(conds[j], _ra[j])

_ra = _ra[task_inds]    
metrics.update('mRA (task)', np.mean(_ra))    
log = metrics.result()
print('Printing {} performance metrics...'.format(args.mode)) 
print(log)
    
np.save(os.path.join(args.output_dir, args.exp_name, args.mode, 'feats.npy'), feats)
print('Done saving {} features!'.format(args.mode))
