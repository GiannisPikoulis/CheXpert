from torch import nn
import torch
import torchvision
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, logger, num_classes, pretrained_imagenet,
                 arch='resnet18', partial_bn=False, extract_features=False):
        
        super(Model, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained_imagenet = pretrained_imagenet
        self.logger = logger
        self._enable_pbn = partial_bn
        self.arch = arch
        self.length = 1
        self.fx = extract_features
        
        if self.arch == 'resnet18' or self.arch == 'resnet34':
            self.num_feats = 512
        elif self.arch == 'resnet50' or self.arch == 'resnet101':
            self.num_feats = 2048
        elif self.arch == 'densenet121':
            self.num_feats = 1024

        self.mean = [0.5031]*3
        self.std = [0.2913]*3
        
        if self.logger:
            self.logger.info(("""
            Initializing CNN with the following configuration:
            CNN Backbone:      {}
            Partial BN:        {}
            """.format(self.arch, self._enable_pbn)))
        
        """
        Construct network
        """
        if self.arch != 'mlp':
            self.net = self._construct_model(arch=self.arch, pretrained=self.pretrained_imagenet)
            self._prepare_layers() 
        else:
            self.net = nn.Sequential(
                nn.Linear(224 * 224, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, self.num_classes)
            )
                
    def _prepare_layers(self):
        
        std = 0.001         
        if self.arch == 'resnet18' or self.arch == 'resnet50' or self.arch == 'densenet121':
            self.classifier = nn.Linear(self.num_feats, self.num_classes)
            torch.nn.init.normal_(self.classifier.weight, 0, std)
            torch.nn.init.constant_(self.classifier.bias, 0)        
        elif self.arch == 'alexnet':
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.num_classes),
            )   
        elif self.arch == 'vgg11_bn':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, self.num_classes),
            )      
    
    def train(self, mode=True):
           
        # Override the default train() to freeze the BN parameters
    
        super(Model, self).train(mode)
        if self._enable_pbn:
            count = 0
            self.logger.info("Freezing BatchNorm2D modules except the first one in CNN model...")
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False      

    def get_optim_policies(self):
        params = [{'params': self.parameters()}]
        return params
     
    def forward(self, x):
        #print(x.shape)                 
        if self.fx:
            x = self.net(x)
            if self.arch == 'densenet121':
              x = F.adaptive_avg_pool2d(x, (1, 1))  
            x = x.flatten(start_dim=1, end_dim=3)
            out = self.classifier(x)
            #print(x.shape, out.shape)
            return out, x
        else: 
            if self.arch != 'mlp':   
                if self.arch == 'densenet121':
                    x = self.net(x)
                    x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
                else:              
                    x = self.net(x).flatten(start_dim=1, end_dim=3)     
                out = self.classifier(x)
                return out
            else:
                x = x[:, 0, :, :]
                x = x.view(x.shape[0], -1)
                out = self.net(x)
                return out 

    def _construct_model(self, arch, pretrained):
        
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        
        model = getattr(torchvision.models, arch)(pretrained = pretrained)
        if pretrained:
            if self.logger:
                self.logger.info('Initializing CNN model with ImageNet pretrained weights...')
       
        modules = list(model.children())[:-1] # delete the last fc layer.
        base_model = nn.Sequential(*modules)            
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(3 * self.length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        
        return base_model

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.net.parameters(), 'lr': lr*lrp},
            {'params': self.fc.parameters(), 'lr': lr}
        ]        
