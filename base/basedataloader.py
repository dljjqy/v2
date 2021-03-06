'''
The truth in torch.utils.DataLoader:
1. The method it used to generate a batch is just simplly concate n tensors.
2. 1 remind me if you want package data in your way, you must write a collect_fn.
'''
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *

class BaseDataLoader(DataLoader):
    '''
    DataLoader desgin for map-style datasets.
    Essentially, dataloader is a iterator for a iterable dataset. 
    '''
    def __init__(self, dataset, batch_size, shuffle, num_workers, sampler=None):
        '''
        BaseDataLoader(dataset, batch_size, shuffle, num_workers, sampler=None, **kwargs)
        kwargs(dict): you can customize parameters which are pass to torch.DataLoader .
        '''
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        self.nclasses = self.dataset.num_classes
        self.batch_size = batch_size
        self.batch_sampler = None
        self.sampler = sampler

        self.init_kwargs = {
            'dataset': self.dataset, 
            'batch_size':batch_size, 
            'shuffle':self.shuffle, 
            'sampler':self.sampler, 
            'batch_sampler':self.batch_sampler, 
            'num_workers':num_workers, 
            'collate_fn':None,
            'pin_memory':True,
            'drop_last':False,
            'timeout':0,
            'worker_init_fn':None
        }

        super(BaseDataLoader, self).__init__(**self.init_kwargs)

class DataPrefetcher(object):
    '''
    Aim: Return a customized iterable object for dataloader, safer and more functionality,
         which is easily turn to a iterator.
    If you simply iter(dataloader) which may raise StopIterationError
    '''
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_pair = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_pair = next(self.loaditer)
        except StopIteration:
            self.next_pair = None
            return 
        with torch.cuda.stream(self.stream):
            self.next_pair._replace(original=self.next_pair.original.cuda(device=self.device, non_blocking=True),
                                    labeled=self.next_pair.labeled.cuda(device=self.device, non_blocking=True))

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_pair is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            pair = self.next_pair
            self.preload()
            count += 1
            yield pair
            if type(self.stop_after) is int and (count > self.stop_after):
                break