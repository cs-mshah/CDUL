import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.datasets import VOCDetection
from torch.utils.data.dataloader import DataLoader
import clip
import lovely_tensors as lt
lt.monkey_patch()
from utils import VOCLabelTransform, CLIPCache


def main():
    voc_target_transform = VOCLabelTransform(transform_type='filename')
    _, preprocess = clip.load('RN50x64')
    dataset = VOCDetection(root=os.environ['DATASETS_ROOT'], year = '2012',image_set='val', transform=preprocess, target_transform=voc_target_transform)
    
    clip_cache = CLIPCache(dataset, thresh=0.5, batch_size=1)
    clip_cache.save(mode='global')

if __name__=='__main__':
    main()