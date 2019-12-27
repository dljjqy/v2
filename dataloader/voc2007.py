import torch
import xmltodict
import numpy as np
from base import BaseDataLoader
from base import BaseDataSet, Pair
from pathlib import Path
from PIL import Image
from torchvision import transforms as transforms
from utils import pad_zero, get_notation

root = "/home/zxpwhu/workspace_jjq/data/VOC2007"

# test_xml = Path(root)/"Annotations/000005.xml"


class VOC2007_Dataset(BaseDataSet):

    def __init__(self, root, train=True, seg=False, size=(500,500)):
        self.seg = seg
        self.train = train
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]
        self.classes = ('person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 
                    'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor')

        self.num_classes = len(self.classes)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])

        # self.transform_for_classification = transforms.Compose([
        # transforms.Resize(size),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=self.MEAN, std=self.STD)
        # ])
        # self.transform_for_seg = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=self.MEAN, std=self.STD)
        # ])
        super(VOC2007_Dataset, self).__init__(root)

        self.size = size
        
    def _correspond(self):
        img_path = self.root/Path("JPEGImages")

        if self.seg:
            lay_out_path = Path(self.root)/"ImageSets/Segmentation/"
            mask_path = Path(self.root)/"SegmentationClass"
            if self.train:
                txt = lay_out_path/"train.txt"
            else:
                txt = lay_out_path/"val.txt"
            with open(txt) as t:
                self.files = [Pair(img_path/(line.split()[0] + '.jpg'), mask_path/(line.split()[0] + '.png')) for line in t.readlines()]
        
        else:
            lay_out_path = Path(self.root)/"ImageSets/Layout"
            xml_path = self.root/Path("Annotations")
            if self.train:
                txt = lay_out_path/Path("train.txt")
            else:
                txt = lay_out_path/Path("val.txt")
            with open(txt) as t:
                self.files = [Pair(img_path/(line.split()[0] + '.jpg'), xml_path/(line.split()[0] + '.xml')) for line in t.readlines()]

        
        
    def _load_data(self, index):
        '''
        The type of image and mask is Image object, label is int between 0-19.
        '''
        assert  index < len(self.files) ,'index must less than the nbr of image'
        if self.seg:
            # print(f'\n\nimage: {self.files[index].image}\n mask: {self.files[index].mask}\n\n')
            
            image = Image.open(self.files[index].image)
            mask = Image.open(self.files[index].mask)
            
            return Pair(self.transform(pad_zero(image,self.size)), 
                        self.transform(pad_zero(mask,self.size)))
        
        else:
            image = Image.open(self.files[index].image)
            data = get_notation(self.files[index].mask)
            #print(f'\n\nimage: {self.files[index].image}\nlabel: {self.files[index].mask}\n\n')
            labels = [0] * self.num_classes
            if isinstance(data['annotation']['object'], list):                 
                for dic in data['annotation']['object']:
                    idx = self.classes.index(dic.get('name'))
                    labels[idx] = 1
            else:
                labels[self.classes.index(data['annotation']['object']['name'])] = 1
            #print(f'\n\n{labels}\n\n')
            return Pair(transforms.ToTensor()(image), torch.Tensor(labels))
        pass
    
class VOC2007_DataLoader(BaseDataLoader):
    
    def __init__(self, dataset, batch_size, shuffle, num_workers, sampler=None):
        super(VOC2007_DataLoader, self).__init__(dataset, batch_size, shuffle, num_workers, sampler)
        self.nclass = dataset.num_classes
        self.classes = dataset.classes

 
if __name__ == "__main__":
    # # test for classification, DONE
    # datset_classification = VOC2007_Dataset(root, seg=False)
    # print(f'Nbr of classification dataset is {len(datset_classification)}')
    # data_iter = iter(datset_classification)
    # image, label = data_iter.__next__()
    # image.show()
    # print(label) 
    
    
    # # test for seg, DONE
    # dataset_seg = VOC2007_Dataset(root, seg=True)
    # print(f'Nbr of classification dataset is {len(dataset_seg)}')
    # data_iter = iter(dataset_seg)
    # image, label = data_iter.__next__()
    # image.show()
    # label.show()

    # # test dataloader, DONE
    # dataset = VOC2007_Dataset(root, seg=True)
    # loader = VOC2007_DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    # print(loader.dataset)
    # print(len(loader))
    # print(len(dataset)//loader.batch_size + 1)

    
    pass