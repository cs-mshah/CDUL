import os
from typing import List
from tqdm import tqdm
from torch import Tensor
import torch
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, InterpolationMode, Normalize, Compose, CenterCrop, Resize
import lovely_tensors as lt
lt.monkey_patch()


class ResNet101Transforms:
    def __init__(self, resize_size=232, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transforms = Compose([
            Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(crop_size),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, img):
        return self.transforms(img)


class TileCropDataset(Dataset):
    def __init__(self, dataset: Dataset, tile_size:tuple = (3, 3)):
        self.dataset = dataset
        self.tile_size = tile_size

    def __len__(self):
        return len(self.dataset)

    def generate_tiles(self, img):
        # Get image dimensions
        img_height, img_width = img.shape[1:]

        # Calculate number of tiles along each dimension
        num_tiles_height = img_height // self.tile_size[0]
        num_tiles_width = img_width // self.tile_size[1]

        # Initialize list to store tiles
        tiles = []

        # Iterate over tiles and store them in a list
        for i in range(num_tiles_height):
            for j in range(num_tiles_width):
                # Calculate tile coordinates
                start_h = i * self.tile_size[0]
                end_h = start_h + self.tile_size[0]
                start_w = j * self.tile_size[1]
                end_w = start_w + self.tile_size[1]

                # Crop the tile
                tile = img[:, start_h:end_h, start_w:end_w]

                # Append the tile to the list
                tiles.append(tile)

        return torch.stack(tiles)
    
    def __getitem__(self, idx):
        original_image, target = self.dataset[idx]
        original_image = ToTensor()(original_image)
        tiles = self.generate_tiles(original_image)
        return (tiles, target)


class CLIPCache:
    def __init__(self, dataset: Dataset, object_categories: List[str], global_cache_dir: str, aggregate_cache_dir: str, thresh: float = 0.5, temperature:float = 1, snippet_size: int = 3, clip_model: str = 'RN50x64', batch_size:int = 16, num_workers:int = 16, device: str = 'cuda'):
        """CLIPCache: Cache vectors after passing images through CLIP
        Args:
            dataset (Dataset): dataset object
            object_categories (List[str]): categories in dataset (labels)
            global_cache_dir (str): directory to save global CLIP cached vectors
            aggregate_cache_dir (str): directory to save aggregate CLIP cached vectors
            thresh (float): threshold parameter (sec. 3.1.3)
            snippet_size (int): size of snippet image
            clip_model (str): CLIP vision encoder model used
            batch_size (int): batch size for processing through CLIP
            num_workers (int): workers for the dataloader
            device (str): device to run CLIP on
        """
        self.dataset = dataset
        self.object_categories = object_categories
        self.global_cache_dir = global_cache_dir
        self.aggregate_cache_dir = aggregate_cache_dir
        self.thresh = thresh
        self.temperature = temperature
        self.snippet_size = snippet_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        # Sec 3.1.3
        self.alpha = None
        self.beta = None
        
        self.model, self.preprocess = clip.load(clip_model, device)
        self.model = self.model.float() # IMPORTANT: https://github.com/openai/CLIP/issues/144
        
        self.text_features = self._text_encode()
        os.makedirs(global_cache_dir, exist_ok=True)
        os.makedirs(aggregate_cache_dir, exist_ok=True)
        
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
        similarity_matrix = (image_features @ self.text_features.T)
        return F.softmax(similarity_matrix / self.temperature, dim=-1)

    def save(self, mode: str = 'global'):
        """save CLIP encoded vectors
        """
        if mode == 'global':
            self._save_global()
        else:
            self._save_aggregate()
        
    def _save_global(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for _, (images, filenames) in enumerate(tqdm(dataloader, desc="Caching CLIP Similarity vectors")):
            images = images.to(self.device)
            similarity = self._image_encode(images)
            for i, filename in enumerate(filenames):
                file_save = os.path.basename(filename).split('.')[0] + '.pt'
                save_tensor = similarity[i].squeeze()
                torch.save(save_tensor, os.path.join(self.global_cache_dir, file_save))
        
    def _save_aggregate(self):
        # remove PIL related transforms
        self.preprocess.transforms.pop(2)
        self.preprocess.transforms.pop(2)
        for i in tqdm(range(len(self.dataset)), desc="Processing dataset"):
            tiles, filename = self.dataset[i]
            similarity = self._compute_in_batches(tiles)
            file_save = os.path.basename(filename).split('.')[0] + '.pt'
            save_tensor = similarity.squeeze()
            torch.save(save_tensor, os.path.join(self.aggregate_cache_dir, file_save))
    
    def _compute_in_batches(self, images: Tensor) -> Tensor:
        """compute similarity vectors for tiles of an image (*, C, snippet_size, snippet_size)
        """
        # reset alpha and beta for each image
        self._reset_alpha_beta()

        for start_idx in tqdm(range(0, images.shape[0], self.batch_size), desc="Processing dataset tiles"):
            end_idx = start_idx + self.batch_size
            
            # pass tiles in batch_size through clip image encoder
            batch_tiles = self.preprocess(images[start_idx:end_idx])
            batch_tiles = batch_tiles.to(self.device)
            tiles_similarity = self._image_encode(batch_tiles)
            self.alpha = torch.max(self.alpha, tiles_similarity.max(dim=0, keepdim=True).values)
            self.beta = torch.min(self.beta, tiles_similarity.min(dim=0, keepdim=True).values)
        
        # Sec 3.1.3 Eq. (5)
        gamma = (self.alpha >= self.thresh).float()
        
        # Sec 3.1.3 Eq. (6)
        return self.alpha * gamma + self.beta * (1 - gamma)
    
    def _reset_alpha_beta(self):
        """reset alpha and beta
        """
        self.alpha = torch.zeros([1, len(self.object_categories)], dtype=torch.float32, device=self.device)
        self.beta = torch.ones([1, len(self.object_categories)], dtype=torch.float32, device=self.device)
    
    def _print_topk(self, similarity: Tensor, k: int = 5):
        """print topk categories, given the similarity vector
        """
        values, indices = similarity.topk(k)
        print(f"\nTop {k} predictions:\n")
        for value, index in zip(values, indices):
            print(f'index: {index}, {self.object_categories[index]}: {100 * value.item():.2f}% ')