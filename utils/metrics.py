import numpy as np
import torch

def miou_duble_classify(pred, target):
    '''
    Double classify means there are only two possible output : 0 or 1.
    '''
    intersection = (pred == target)
    union = (pred + target - intersection)
    return intersection.sum().item() / union.sum().item()

def pixelAcc_double_classify(pred, target):
    intersection = (pred == target)
    return intersection.sum().item() / (pred.sum().item() + np.spacing(1))

def batch_miou(output, target, num_classes):
    output = output.long()
    intersection = output * (output == target).long()

    inter = torch.histc(intersection.float(), bins=num_classes, min=1, max=num_classes)
    pred = torch.histc(output.float(), bins=num_classes, min=1, max=num_classes)
    label = torch.histc(target.float(), bins=num_classes, min=1, max=num_classes)
    
    union = pred + label - inter
    assert (inter <= union).all() , 'Intersection area should smaller than Union area'
    return (inter.cpu().numpy() / (union.cpu().numpy() + np.spacing(1))).sum() / num_classes 

def batch_pixelAcc(output, target, num_classes):
    output = output.long()
    intersection = output * (output == target).long()

    pixel_correct = torch.histc(intersection.float(), bins=num_classes, min=1, max=num_classes)
    pixel_labeled = torch.histc(output.float(), bins=num_classes, min=1, max=num_classes)
    
    return pixel_correct.sum().cpu().numpy() /  (pixel_labeled.sum().cpu().numpy() + np.spacing(1))
    