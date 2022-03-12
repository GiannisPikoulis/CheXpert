import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, make_barplot
import model.metric
import model.loss


class Trainer(BaseTrainer):
    
    def __init__(self, model, criterion, metric, optimizer, 
                 config, train_dataloader, val_dataloader, lr_scheduler=None, len_epoch=None):
        
        super().__init__(model, config)
            
        self.config = config    
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.do_validation = self.val_dataloader is not None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.conditions = self.val_dataloader.dataset.conditions
        
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(train_dataloader)
            self.len_epoch = len_epoch
        
        self.log_step = int(self.len_epoch / 10)    
        self.metric = metric
        self.criterion = criterion
        self.task_inds = [2, 5, 6, 8, 10]
        self.train_metrics = MetricTracker('Loss', 'mAP', 'mRA', "mRA (task)", "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", writer=self.writer)
        self.val_metrics = MetricTracker('Loss', 'mAP', 'mRA', "mRA (task)", "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", writer=self.writer)     

    def _train_epoch(self, epoch, phase="train"):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metrics in this epoch.
        """
        
        if phase == "train": 
            self.logger.info("Starting training phase for epoch: {}".format(epoch)) 
            self.logger.info("Printing learning rates...")
            for param_group in self.optimizer.param_groups:
                self.logger.info(param_group['lr'])       
            self.model.train()
            self.train_metrics.reset()
            torch.set_grad_enabled(True)
            metrics = self.train_metrics
        
        elif phase == "val":
            self.logger.info("Starting validation phase for epoch: {}".format(epoch))
            self.model.eval()
            self.val_metrics.reset()
            torch.set_grad_enabled(False)            
            metrics = self.val_metrics

        _outputs = []
        _targets = []
        
        running_loss = 0
        total_loss = 0
        
        dataloader = self.train_dataloader if phase == "train" else self.val_dataloader
        
        for batch_idx, batch_data in enumerate(dataloader):
                     
            X = batch_data[0].to(self.device) 
            Y = batch_data[1].to(self.device) 
            
            if phase == "train":
                self.optimizer.zero_grad()
                
            out = self.model(X)
                
            loss = 0
            loss += self.criterion(out, Y)

            if phase == "train":
                loss.backward()
                self.optimizer.step() 
                
            running_loss += loss.item()    
            total_loss += loss.item()
            
            output = out.cpu().detach().numpy()
            targ = Y.cpu().detach().numpy()
            _outputs.append(output)
            _targets.append(targ)

            if (batch_idx % self.log_step == self.log_step - 1) and phase == 'train':
                self.logger.info('[Epoch: {}] {} [Running Loss: {:.4f}]'.format(epoch,
                                 self._progress(batch_idx), running_loss / self.log_step))
                running_loss = 0

            if batch_idx == self.len_epoch and phase == 'train':
                break
            
        self.writer.set_step(epoch, phase)

        if phase == 'val':
            metrics.update('Loss', total_loss / len(dataloader))
        else:
            metrics.update('Loss', total_loss / self.len_epoch)
            
        out_cat = np.vstack(_outputs)
        target_cat = np.vstack(_targets)

        conditions = self.conditions
	
        if phase == 'val':
            out_cat = out_cat[:, [x for x in range(14) if x != 12]]
            target_cat = target_cat[:, [x for x in range(14) if x != 12]]
            conditions = self.conditions[:12] + [self.conditions[13]] 

        _ap = model.metric.average_precision(out_cat, target_cat)
        _ra = model.metric.roc_auc(out_cat, target_cat)
        metrics.update("mAP", np.mean(_ap))
        metrics.update("mRA", np.mean(_ra))
        for j in range(len(conditions)):
            metrics.update(conditions[j], _ra[j])

        _ra = _ra[self.task_inds]
        metrics.update("mRA (task)", np.mean(_ra))
        
        log = metrics.result()
         
        if phase == "train":
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.do_validation:
                val_log = self._train_epoch(epoch, phase="val")
                log.update(**{'Validation ' + k: v for k, v in val_log.items()})
            
            return log

        elif phase == "val":
            self.writer.save_results(out_cat, "out_cat")
            return metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_dataloader, 'n_samples'):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
