import hydra
import copy
import os
from omegaconf import DictConfig, OmegaConf
import rootutils
import wandb
from loguru import logger as log
from torchvision.models import ResNet101_Weights
from torch.utils.data import DataLoader
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MultilabelAveragePrecision
import lovely_tensors as lt
lt.monkey_patch()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.rich_utils import print_config_tree
from src.utils.utils import set_seed


class Trainer:
    def __init__(self, cfg):
        object_categories = hydra.utils.instantiate(cfg.data.object_categories)
        train_transform = val_transform = ResNet101_Weights.transforms
        
        # for defining the initial pseudo labels
        target_transform = hydra.utils.instantiate(cfg.data.target_transform, 
                                                object_categories=object_categories,
                                                transform_type=cfg.train.target_transform, 
                                                global_cache_dir=cfg.clip_cache.global_cache_dir,
                                                aggregate_cache_dir=cfg.clip_cache.aggregate_cache_dir)
        
        # used in the val dataset
        onehot_transform = hydra.utils.instantiate(cfg.data.target_transform, 
                                                object_categories=object_categories,
                                                transform_type='onehot')
        
        # train dataset with targets as (initial pseudo labels ('final', 'global', 'aggregate'), onehot ground truth)
        train_dataset = hydra.utils.instantiate(cfg.data.dataset, transform=train_transform, target_transform=target_transform)
        
        val_dataset_cfg = copy.deepcopy(cfg.data.dataset)
        val_dataset_cfg.image_set = 'val'
        val_dataset = hydra.utils.instantiate(val_dataset_cfg, transform=val_transform, target_transform=onehot_transform)
        
        # IMPORTANT: keep persistent_workers=False to change psuedo labels on the fly
        self.train_dataloader = DataLoader(train_dataset, 
                                    batch_size=cfg.train.get("batch_size", 8), 
                                    shuffle=True, num_workers=cfg.train.get("num_workers", 4), 
                                    persistent_workers=False)
        self.val_dataloader = DataLoader(val_dataset, 
                                    batch_size=cfg.train.get("batch_size", 8), 
                                    shuffle=False, num_workers=cfg.train.get("num_workers", 4), 
                                    persistent_workers=False)
        
        self.model = hydra.utils.instantiate(cfg.train.model)
        self.model = self.model.to(cfg.train.device)
        self.optimizer = hydra.utils.instantiate(cfg.train.optimizer, params=self.model.parameters())
        
        self.train_metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
        self.train_metric = self.train_metric.to(cfg.train.device)
        
        self.val_metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
        self.val_metric = self.val_metric.to(cfg.train.device)
        
        self.loss = hydra.utils.instantiate(cfg.train.loss)
        
    def fit(self):
        raise NotImplementedError


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    print_config_tree(cfg)
    
    # initialise seeds here!
    set_seed(cfg.train.seed)
    
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=os.environ['WANDB_ENTITY'],
               project='CDUL',
               name=f'{cfg.task_name}',
               config=wandb_config,
               id=None, # replace with run id when resuming a run
               resume='allow',
               tags=None,
               allow_val_change=True,
               settings=wandb.Settings(start_method="fork"))
    
    trainer = Trainer(cfg)
    trainer.fit()


if __name__=='__main__':
    main()