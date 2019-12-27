#import logging
import torch
import utils
#import json
from torch.utils import tensorboard
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# def creat_object(module, name, **args):
#     return getattr(module, name)(**args)

class BaseTrainer:
    def __init__(self, epochs, device, model, Loss, optimizer, lr_scheduler, trainloader,  
                do_val, val_loader, epochsForVal, tb, epochsForSave,
                save_dir, resume, resume_path):
        
        self.epochs = epochs
        self.device = device
        self.model = model.to(self.device)
        self.Loss = Loss

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.trainloader = trainloader

        self.epochsForVal = epochsForVal
        self.do_val = do_val
        self.val_loader = val_loader
        
        self.epochsForSave = epochsForSave
        
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        if tb:
            self.tb= True
            self.writer = tensorboard.SummaryWriter(str(save_dir))

        self.resume = resume
        self.resume_path = resume_path
        if self.resume: self._resume_checkpoints()

    def train(self):
        for epoch in range(self.epochs):
            if self.do_val and epoch+1 % self.epochsForVal == 0:
                results = self._val_epoch(epoch)
            else:
                results = self._train_epoch(epoch)
            if epoch % self.epochsForSave == 0:
                self._save_checkpoints(epoch)
        pass

    def _train_epoch(self, epoch):
        raise NotImplementedError
    def _val_epoch(self, epoch):
        raise NotImplementedError
    def _eval_metrics(self, image, mask):
        raise NotImplementedError
    def _check_best(self, result):
        raise NotImplementedError

    def _save_checkpoints(self, epoch):
        filename = Path(self.save_dir)/f'-{self.model.__class__.__name__}_{epoch}.pth' if not self._check_best \
            else Path(self.save_dir)/f'{self.model.__class__.__name__}_best.pth'
        
        torch.save(self.model.state_dict(), filename)

    def _resume_checkpoints(self):
        self.model.load_state_dict(torch.load(self.resume_path))