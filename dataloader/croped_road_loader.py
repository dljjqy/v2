from base import BaseDataSet, BaseDataLoader, Pair
from PIL import Image
from pathlib import Path

class Crop_Road_Set(BaseDataSet):
    def __init__(self, root, train=True):
        self.train = train
        self.numclass = 2
        self.classes = ('Unknow', 'Road')

        super(Crop_Road_Set, self).__init__(root)
    
    def _correspond(self):
        paths_iter = self.root.glob('*.jpg')
        files = [Pair(path, self.root/path.name.split('.')[0]+'.png') for path in paths_iter]
        if self.train:
            self.files = files[0:int(0.9*len(files))]
        else:
            self.files = files[int(0.9*len(files)):]

    def _load_data(self, index):
        assert  index < len(self.files) ,'index must less than the nbr of image'
        
        pair = self.files[index]
        return Pair(Image.open(pair.image), Image.open(pair.mask))

class Crop_Road_Loader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, sampler=None):
        super(Crop_Road_Loader, self).__init__(dataset, batch_size, shuffle, num_workers, sampler)
        self.nclass = dataset.num_classes
        self.classes = dataset.classes