import os
from tqdm import tqdm
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MultilabelAveragePrecision
import clip
import rootutils
import lovely_tensors as lt
lt.monkey_patch()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.rich_utils import print_config_tree

@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    
    print_config_tree(cfg)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load(cfg.model.clip.name, device)
    
    object_categories = hydra.utils.instantiate(cfg.data.object_categories)

    predict_transform = hydra.utils.instantiate(cfg.data.target_transform, 
                                                object_categories=object_categories,
                                                transform_type=cfg.evaluate.mode, 
                                                global_cache_dir=cfg.clip_cache.global_cache_dir,
                                                aggregate_cache_dir=cfg.clip_cache.aggregate_cache_dir)
    
    predicted_dataset = hydra.utils.instantiate(cfg.data.dataset, transform=preprocess, target_transform=predict_transform)
    
    predict_dataloader = DataLoader(predicted_dataset, batch_size=cfg.evaluate.get("batch_size", 8), 
                                    shuffle=False, num_workers=cfg.evaluate.get("num_workers", 4))
    
    metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
    metric = metric.to(device)
    
    for (_, (pred_labels, target_labels)) in tqdm(predict_dataloader, desc="Processing batch"):
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