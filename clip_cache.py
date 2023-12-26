import os
from torchvision.datasets import VOCDetection
import clip
import lovely_tensors as lt
lt.monkey_patch()
from utils import VOCLabelTransform, TileCropDataset, CLIPCache, get_categories


def main():
    
    object_categories = get_categories(os.path.join(os.environ['DATASETS_ROOT'], 'VOCdevkit/VOC2012/ImageSets/Main'))
    voc_target_transform = VOCLabelTransform(transform_type='filename')
    _, preprocess = clip.load('RN50x64')
    save_root = os.path.join(os.environ['DATASETS_ROOT'], 'VOCdevkit/VOC2012/')
    
    dataset = VOCDetection(root=os.environ['DATASETS_ROOT'], year = '2012',image_set='val', transform=preprocess, target_transform=voc_target_transform)
    
    clip_cache = CLIPCache(dataset, object_categories, save_root, batch_size=8)
    
    # Section 3.1.1: save soft global similarity vectors
    clip_cache.save(mode='global')
    
    dataset = VOCDetection(root=os.environ['DATASETS_ROOT'], year='2012', image_set='val', target_transform=voc_target_transform)
    tile_crop_dataset = TileCropDataset(dataset, tile_size=(3, 3))
    
    # Section 3.1.2, 3.1.3: save soft aggregation similarity vectors
    clip_cache = CLIPCache(tile_crop_dataset, object_categories, save_root, batch_size=8)
    clip_cache.save(mode='aggregate')

if __name__=='__main__':
    main()