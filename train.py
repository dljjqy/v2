import dataloader
import torch
from net.pspnet import PSPNet 
from pathlib import Path
from utils.lr_scheduler import poly_lr
from utils.loss import *
from trainer import Trainer_for_seg
'''
This python file aims to start training by initilize a Trainer object.

'''
trainset_kwargs = {
    'root':"/home/zxpwhu/workspace_jjq/data/voc2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012",
    'train':True,
    'seg':True
}

valset_kwargs = {
    'root':"/home/zxpwhu/workspace_jjq/data/voc2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012",
    'train':False, 
    'seg':True
}
trainset = getattr(dataloader, 'VOC2012_Dataset')(**trainset_kwargs)
valset = getattr(dataloader, 'VOC2012_Dataset')(**valset_kwargs)
valset = None,

trainloader_kwargs = {
    'dataset':trainset,
    'batch_size':8,
    'shuffle':True,
    'num_workers':4
}
valloader_kwargs = {
    'dataset':valset,
    'batch_size':1, 
    'shuffle':False,
    'num_workers':4
} 

trainloader = getattr(dataloader, 'VOC2012_DataLoader')(**trainloader_kwargs)
valloader = None
testloader = None

model = PSPNet(trainloader.nclass)
cerition = nn.CrossEntropyLoss()
params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params=params)
lr_scheduler = poly_lr()

trainer_kwargs = {
    'epochs': 5,
    'device':torch.device('cuda:0'),
    'model':model,
    'Loss':cerition,
    'optimizer':optimizer,
    'lr_scheduler':lr_scheduler,
    'trainloader': trainloader,
    
    'do_val': True, 
    'val_loader': valloader,
    'epochsForVal':5,

    'tb': True, 
    
    'epochsForSave':1,
    'save_dir':'/home/zxpwhu/workspace_jjq/v2/runs',
    
    'resume':False,
    'resume_path':''
}

torch.backends.cudnn.benchmark = True
trainer = Trainer_for_seg(**trainer_kwargs)

if __name__ == "__main__":
    trainer.train()
    pass