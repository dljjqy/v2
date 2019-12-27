import torch
import xmltodict
from PIL import Image
import numpy as np
'''
This is a module contains many funciton which may be used in many other module.
Tips:
    Original image size : Width X Height
    PIL.Image().size -> Width X Height
    toTensor(img).shape -> Channels X Height X Width
    np.asarray(img) -> Height X Width X Channels
'''
class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = 0
        self.initilized = False

    def update(self, val, weight=1):
        if not self.initilized:
            self.val=val
            self.sum = np.multiply(val, weight)
            self.cnt = weight
        else:
            self.initilized = True
            self.val = val
            self.sum = np.add(self.sum, np.multiply(val, weight))
            self.cnt += weight
    
    @property
    def average(self):
        return np.round(self.sum/self.cnt, 5)

def pad_zero(image, size):
    '''
    If the images in dataset holds different size, dataloader can not pack some images
    into a batch.But if the size of image in this dataset has limition, 
    maybe smaller tha the "size" then we can pad black pixel around the image until 
    its size become "size". 
    '''
    size_x, size_y = size   # size_x -> width, size_y -> height
    # Check type 
    if isinstance(image, Image.Image):
        width, height = image.size
        assert size_x >= width and size_y >= height
        new_img = Image.new(image.mode, size, 0)
        start_x, start_y = (size_x-width)//2, (size_y-height)//2
        new_img.paste(image, (start_x,start_y))

    elif isinstance(image, np.ndarray):
        old_shape = image.shape
        new_shape = (*size[::-1], *old_shape[2:])
        height, width = old_shape[0], old_shape[1]
        assert size_x >= width and size_y >= height
        new_img = np.zeros(new_shape, dtype=np.uint8)
        start_x, start_y = (size_x-width)//2, (size_y-height)//2
        new_img[start_y:start_y+height, start_x:start_x+width] = image
        
    elif isinstance(image, torch.Tensor):
        width, height = image.shape[-1], image.shape[-2]
        new_shape = (*image.shape[-3::-1], *size[::-1])
        assert size_x >= width and size_y >= height
        new_img = torch.zeros(new_shape)
        start_x, start_y = (size_x-width)//2, (size_y-height)//2
        new_img[:, start_y:start_y+height, start_x:start_x+width] = image
    
    return new_img

def get_notation(xml_path):
    xml_file = open(xml_path, 'r')
    xml = xml_file.read()
    json = xmltodict.parse(xml)
    return json

def denormalize(tensors, mean, std):
    for tensor in tensors:
        for c, m, s in zip(tensor, mean, std):
            c.mul_(s).add_(m)

if __name__ == "__main__":
    # test has passed
    from torchvision import transforms as transforms
    test_img = Image.open('/home/jjq/000012.jpg')
    test_img_np = np.asarray(test_img)
    test_img_tensor = transforms.ToTensor()(test_img)
    
    pad_img = pad_zero(test_img,(500,500))
    pad_img_np = pad_zero(test_img_np,(500,500))
    pad_img_tensor = pad_zero(test_img_tensor,(500,500))

    print(f'pad_image:{type(pad_img)} size:{pad_img.size}')
    print(f'pad_image_np:{type(pad_img_np)} shape:{pad_img_np.shape}')
    print(f'pad_image_tensor:{type(pad_img_tensor)} shape:{pad_img_tensor.shape}')

    pad_img_np = Image.fromarray(pad_img_np)
    pad_img_tensor = transforms.ToPILImage()(pad_img_tensor)
    print(type(pad_img_tensor))
    pad_img.show()
    pad_img_np.show()
    pad_img_tensor.show()

   