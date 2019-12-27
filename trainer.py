import torch
import numpy as np
import torch.nn as nn
from base import BaseTrainer
from utils import AverageMeter
from utils.metrics import * 
from collections import namedtuple
from torchvision import transforms as transforms
from time import time

# Results = namedtuple('Results', ['loss', 'miou', 'pixelAcc'])


class Trainer_for_seg(BaseTrainer):
    def __init__(self, mode='train',**kwargs):
        super(Trainer_for_seg, self).__init__(**kwargs)
        self.best = {'loss':np.inf, 'miou':0, 'pixelAcc':0}
        self.total_iter = len(self.trainloader) * self.epochs
        self.mode = mode

    def _train_epoch(self, epoch):
        self.init_metrics()
        self.model.train()

        for idx, (data, mask) in enumerate(self.trainloader):
            itr = 1 + idx + epoch * len(self.trainloader)
            tic = time()

            data, mask = data.to(self.device), mask.to(self.device)
            output = self.model(data)[-1]
            
            # Loss and optimize
            lr = self.lr_scheduler(itr, self.total_iter)
            self.optimizer.zero_grad()

            mask=mask.long()
            output = nn.functional.softmax(output, dim=1)
            # print(f'\n### {output.shape} ###\n')
            # print(f'\n### {mask.shape} ###\n')

            loss = self.Loss(output, mask)
            self.total_loss.update(loss.item())
            loss.backward()
            self.optimizer.lr = 0.0001
            self.optimizer.step()

            # Calculate metrics
            output = torch.argmax(output, dim=1)
            miou, pixelAcc = self._eval_metrics(output, mask)
            self.miou.update(miou)
            self.pixelAcc.update(pixelAcc)
            
            # elapsed time
            elapsed_time = time() - tic
            # print information
            # This can be turn to tabr
            print(f'epoch:{epoch+1:3d}, iteration:{itr:4d}, loss:{loss.item():.3f}, miou:{miou:.3f}, pixelAcc:{pixelAcc:.2f}, lr:{self.optimizer.lr:.6f},Time:{elapsed_time:.2f}')
        
            # Tensorboard
            if self.tb:
                self.writer.add_scalar('trainint_loss', self.total_loss.val, itr)
                self.writer.add_scalar('miou', self.miou.val, itr)
                self.writer.add_scalar('pixelAcc', self.pixelAcc.val, itr)
                self.writer.add_scalar('learning_rate', lr, itr)

                for i, opt_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'{self.mode}/learning_rate_{i}', opt_group['lr'], itr)
            

            # check best or not
            results = {'loss':self.total_loss.average, 'miou':self.miou.average, 
                        'pixelAcc':self.pixelAcc.average}
            if self._check_best(results):
                self._save_checkpoints(epoch+1)
        
        return results 
            

    def _val_epoch(self, epoch):
        self.init_metrics()

        with torch.no_grad():
            for idx, (data, mask) in enumerate(self.trainloader):
                itr = 1 + idx + epoch * len(self.trainloader)
                data, mask = data.to(self.device), mask.to(self.device)

                output = self.model(data)
                loss = self.Loss(output, mask)

                self.total_loss.update(loss.item)
                miou, pixelAcc = self._eval_metrics(output, mask)
                self.miou.update(miou)
                self.pixelAcc.update(pixelAcc)

                print(f'epoch:{epoch}, loss:{loss.item():.2f}, miou:{miou}, pixelAcc:{pixelAcc}')
        return {'loss':self.total_loss.average, 'miou':self.miou.average, 
                        'pixelAcc':self.pixelAcc.average} 

    def init_metrics(self):
        self.time = AverageMeter()
        self.total_loss = AverageMeter()
        self.miou = AverageMeter()
        self.pixelAcc = AverageMeter()
        self.model.eval()


    def _check_best(self, results):
        if results['loss'] < self.best['loss'] \
            or results['miou'] >= self.best['miou'] \
            or results['pixelAcc'] >= self.best['pixelAcc']:
            self.best = results
            return True
        else:
            return False  

    def _eval_metrics(self, images, masks):
        num_class = self.trainloader.nclasses
        miou = batch_miou(images, masks, num_class)
        pixAcc = batch_pixelAcc(images, masks, num_class)
        
        return miou, pixAcc