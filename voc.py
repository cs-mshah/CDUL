import os
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.datasets import VOCDetection
from torch.utils.data.dataloader import DataLoader
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MultilabelAveragePrecision
import clip
import lovely_tensors as lt
lt.monkey_patch()

from utils import get_categories, VOCLabelTransform


def main():
    
    # indexing depends of index of label in object_categories
    object_categories = get_categories(os.path.join(os.environ['DATASETS_ROOT'], 'VOCdevkit/VOC2012/ImageSets/Main'))
    print(object_categories)
    target_transform = VOCLabelTransform(object_categories, exclude_difficult=False)
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50x64', device)
    model = model.float() # important: https://github.com/openai/CLIP/issues/144
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in object_categories]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    test_dataset = VOCDetection(root=os.environ['DATASETS_ROOT'], year = '2012',image_set='val', transform=preprocess, target_transform=target_transform)
    
    batch_size = 8
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
    metric = metric.to(device)
    
    for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader, desc="Processing batch")):
        image_input = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            image_features: Tensor = model.encode_image(image_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ text_features.T)
        temperature = 1.0
        similarity = F.softmax(similarity_matrix / temperature, dim=-1)
        metric.update(similarity, labels)

    
    print(f'class_ap: {metric.compute()}')
    metric.reset()

if __name__=='__main__':
    main()