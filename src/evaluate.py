import os
from tqdm import tqdm
import torch
from torchvision.datasets import VOCDetection
from torch.utils.data.dataloader import DataLoader
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MultilabelAveragePrecision
import clip
import lovely_tensors as lt
lt.monkey_patch()

from utils import get_categories, VOCLabelTransform


def main():
    
    object_categories = get_categories(os.path.join(os.environ['DATASETS_ROOT'], 'VOCdevkit/VOC2012/ImageSets/Main'))
    saved_dir = os.path.join(os.environ['DATASETS_ROOT'], 'VOCdevkit/VOC2012', 'clip_cache/RN50x64_3')
    global_tensors_transform = VOCLabelTransform(transform_type='global', saved_dir=saved_dir)
    one_hot_transform = VOCLabelTransform(object_categories)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load('RN50x64', device)
    
    # evaluate the cached global similarity vectors obtained from CLIP
    predicted_dataset = VOCDetection(root=os.environ['DATASETS_ROOT'], year = '2012',image_set='val', transform=preprocess, target_transform=global_tensors_transform)
    target_dataset = VOCDetection(root=os.environ['DATASETS_ROOT'], year = '2012',image_set='val', transform=preprocess, target_transform=one_hot_transform)
    
    batch_size = 8
    predict_dataloader = DataLoader(predicted_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
    metric = metric.to(device)
    
    for (_, pred_labels), (_, target_labels) in tqdm(zip(predict_dataloader, target_dataloader), desc="Processing batch"):
        pred_labels = pred_labels.to(device)
        target_labels = target_labels.to(device)
        metric.update(pred_labels, target_labels)

    AP_per_class = metric.compute()
    metric.reset()
    
    print(f'classwise_ap: {AP_per_class}')
    mAP = torch.mean(torch.tensor(list(AP_per_class.values())))
    print(f'mAP: {mAP.item()}')
    

if __name__=='__main__':
    main()