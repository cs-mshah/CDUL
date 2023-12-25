import os
from typing import Dict, List
from tqdm import tqdm
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import VOCDetection
import clip
from torch.utils.data import DataLoader
import lovely_tensors as lt
lt.monkey_patch()


def get_categories(labels_dir: str) -> List[str]:
    """
    Get the object categories for Pascal VOC 2012

    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a sorted list to maintain consistency
    """

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError

    else:
        categories: List[str] = []

        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])
        categories.sort()
        return categories


class VOCLabelTransform:
    def __init__(self, object_categories: str | None = None, exclude_difficult: bool = False, transform_type: str = 'one_hot', saved_dir: str = None):
        """VOC Label Transform
        Args:
            object_categories (list): List of object categories.
            exclude_difficult (bool): If True, exclude objects labeled as difficult (for one-hot).
            transform_type (VOCLabelTransformType): Type of label transform to apply.
            saved_dir (str): Directory to fetch the saved label tensors.
        """
        self.object_categories = object_categories
        self.exclude_difficult = exclude_difficult
        self.transform_type = transform_type
        self.saved_dir = saved_dir

    def __call__(self, target: Dict):
        """
        return label transform depending on transform_type

        Args:
            target: xml tree file
        Returns:
            target with transform type applied
        """
        if self.transform_type == 'one_hot':
            return self.one_hot(target)
        elif self.transform_type == 'filename':
            return self.filename_label(target)
        elif self.transform_type == 'global':
            return self.global_label(target)
        elif self.transform_type == 'aggregate':
            return self.aggregated_label(target)
        elif self.transform_type == 'final':
            return self.final_label(target)
        

    def one_hot(self, target) -> Tensor:
        """
        return one hot encoded labels
        """
        ls = target['annotation']['object']

        j = []
        if type(ls) == dict:
            if int(ls['difficult']) == 0:
                j.append(self.object_categories.index(ls['name']))
            elif int(ls['difficult']) == 1 and self.exclude_difficult == False:
                j.append(self.object_categories.index(ls['name']))
        elif type(ls) == list:
            for i in range(len(ls)):
                if int(ls[i]['difficult']) == 0:
                    j.append(self.object_categories.index(ls[i]['name']))
                elif int(ls[i]['difficult']) == 1 and self.exclude_difficult == False:
                    j.append(self.object_categories.index(ls[i]['name']))
        else:
            raise TypeError

        k = np.zeros(len(self.object_categories), dtype=int)
        k[j] = 1 # object present

        return torch.from_numpy(k)
    
    def filename_label(self, target) -> str:
        """
        return one hot encoded label
        """
        return target['annotation']['filename']

    
    def global_label(self, target) -> Tensor:
        """returns soft global saved vector
        """
        raise NotImplementedError
    
    def aggregated_label(self, target) -> Tensor:
        """returns soft aggregation saved vector
        """
        raise NotImplementedError
    
    def final_label(self, target) -> Tensor:
        """returns initial pseudo label
        """
        return 0.5 * (self.global_label(target) + self.aggregated_label(target))



class CLIPCache:
    def __init__(self, dataset: VOCDetection, thresh: float, temperature:float = 1, snippet_size: int = 3, clip_model: str = 'RN50x64', batch_size:int = 16, device: str = 'cuda'):
        """CLIPCache: Cache vectors after passing images through CLIP
        Args:
            dataset (VOCDetection): dataset object
            thresh (float): threshold parameter (sec. 3.1.3)
            snippet_size (int): size of snippet image
            clip_model (str): CLIP vision encoder model used
            batch_size (int): batch size for processing through CLIP
            device (str): device to run CLIP on
        """
        self.dataset = dataset
        self.thresh = thresh
        self.temperature = temperature
        self.snippet_size = snippet_size
        self.batch_size = batch_size
        
        self.device = device
        self.model, _ = clip.load(clip_model, device)
        self.model = self.model.float() # IMPORTANT: https://github.com/openai/CLIP/issues/144
        
        self.object_categories = get_categories(os.path.join(self.dataset.root, 'VOCdevkit/VOC2012/ImageSets/Main'))
        
        self.text_features = self._text_encode()
        self.save_root = os.path.join(self.dataset.root, f"VOCdevkit/VOC2012/clip_cache/{clip_model}_{self.snippet_size}")
        os.makedirs(os.path.join(self.save_root, 'global'), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, 'aggregate'), exist_ok=True)
        
    @torch.no_grad()
    def _text_encode(self) -> Tensor:
        """clip encode categories text
        """
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.object_categories]).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def _image_encode(self, image: Tensor) -> Tensor:
        """clip encode image and find similarity with text_features (can have batch dimension)
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity_matrix = (100.0 * image_features @ self.text_features.T)
        return F.softmax(similarity_matrix / self.temperature, dim=-1)

    def save(self, mode: str = 'global'):
        """save CLIP encoded vectors
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        for batch_idx, (images, filenames) in enumerate(tqdm(dataloader, desc="Caching CLIP Similarity vectors")):
            images = images.to(self.device)
            print(f'images: {images.v}')
            if mode == 'global':
                similarity = self._image_encode(images)
                save_path = os.path.join(self.save_root, 'global')
            else:
                similarity = self.generate_snippets(images)
                save_path = os.path.join(self.save_root, 'aggregate')

            for i, filename in enumerate(filenames):
                file_save = os.path.basename(filename).split('.')[0] + '.pt'
                torch.save(similarity[i], os.path.join(save_path, file_save))

    def generate_snippets(self):
        """generate snippets and return aggregate vector
        """
        raise NotImplementedError
    
    def _print_topk(self, similarity: Tensor, k: int = 5):
        """print topk categories, given the similarity vector
        """
        values, indices = similarity.topk(k)
        print(f"\nTop {k} predictions:\n")
        for value, index in zip(values, indices):
            print(f'index: {index}, {self.object_categories[index]}: {100 * value.item():.2f}% ')



def get_label_txt(label: torch.Tensor, object_categories: List) -> List[str]:
    """
    Get Label text from numpy array encoded as 0/1 vector

    Args:
        label: 0/1 encoded label
    Returns:
        List of  categories string
    """
    k = label.numpy()
    labeltext=[]
    for i in range(len(k)):
      if k[i] >= 1:
        labeltext.append(object_categories[i])

    return labeltext
    
    