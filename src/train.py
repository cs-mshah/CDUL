import math
import hydra
import os
from datetime import datetime
import shutil
from tqdm import tqdm
from typing import Tuple
from omegaconf import DictConfig, OmegaConf
import rootutils
import wandb
from loguru import logger as log
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.regression import KLDivergence
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MultilabelAveragePrecision
import lovely_tensors as lt
lt.monkey_patch()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.rich_utils import print_config_tree
from src.utils.utils import set_seed


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        self.sigma: float = cfg.train.sigma # sec 3.2 eq. 8
        self.device: str = cfg.train.device
        object_categories: list = hydra.utils.instantiate(cfg.data.object_categories)
        train_transform = hydra.utils.instantiate(cfg.train.train_transform)
        val_transform = hydra.utils.instantiate(cfg.train.val_transform)
        
        # for defining the initial pseudo labels
        target_transform = hydra.utils.instantiate(
            cfg.data.target_transform,
            object_categories=object_categories,
            transform_type=cfg.train.target_transform,
            global_cache_dir=cfg.clip_cache.global_cache_dir,
            aggregate_cache_dir=cfg.clip_cache.aggregate_cache_dir,
            pseudo_cache_dir=cfg.clip_cache.pseudo_cache_dir,
            final_lambda=cfg.clip_cache.final_lambda,
        )
        
        # used in the val dataset
        onehot_transform = hydra.utils.instantiate(cfg.data.target_transform, 
                                                object_categories=object_categories,
                                                transform_type='onehot')
        
        # train dataset with targets as (initial pseudo labels ('final', 'global', 'aggregate'), onehot ground truth)
        train_dataset = hydra.utils.instantiate(cfg.data.dataset, transform=train_transform, target_transform=target_transform)
        
        val_dataset = hydra.utils.instantiate(cfg.data.val_dataset, transform=val_transform, target_transform=onehot_transform)
        
        self.train_dataloader = DataLoader(train_dataset, 
                                    batch_size=cfg.train.get("batch_size", 8), 
                                    shuffle=True, num_workers=cfg.train.get("num_workers", 4))
        self.val_dataloader = DataLoader(val_dataset, 
                                    batch_size=cfg.train.get("batch_size", 8), 
                                    shuffle=False, num_workers=cfg.train.get("num_workers", 4))
        
        self.model: nn.Module = hydra.utils.instantiate(cfg.model.resnet101, num_labels=len(object_categories))
        self.model = self.model.to(self.device)

        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.train.optimizer, params=self.model.parameters())
        
        self.train_metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
        self.train_metric = self.train_metric.to(self.device)
        
        self.val_metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
        self.val_metric = self.val_metric.to(self.device)
        
        # measure pseudo-label quality over epochs
        self.warmup = cfg.train.get("warmup", 0)
        self.pseudo_update_frequency = cfg.train.get("pseudo_update_frequency", 1)
        self.pseudo_metric = ClasswiseWrapper(MultilabelAveragePrecision(num_labels=len(object_categories), average=None), labels=object_categories)
        self.pseudo_metric = self.pseudo_metric.to(self.device)
        
        self.best_val_mAP = 0.0
        
        self.loss: KLDivergence = hydra.utils.instantiate(cfg.train.loss)
        self.loss = self.loss.to(self.device)
    
    def fit(self) -> float:
        start_epoch = self.resume_checkpoint()
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            total_loss, train_mAP, pseuso_mAP = self.train(epoch)
            stop = self.early_stopping(pseuso_mAP, epoch)
            if stop:
                log.warning(f'early stopping at epoch: {epoch+1} since pseudo mAP is decreasing. Initial: {self.pseudo_mAP_start}, Current: {pseuso_mAP}')
                break
            val_mAP = self.validation(epoch)
            wandb.log({'train/pred_pseudo_kl': total_loss, 'train/mAP': train_mAP, 'train/pseudo_mAP': pseuso_mAP, 'val/mAP': val_mAP})
            log.info(f'[epoch: {epoch+1}] train_loss KL(pred,pseudo): {total_loss}, train_mAP: {train_mAP}, pseudo_mAP: {pseuso_mAP}, val_mAP: {val_mAP}')
            self.save_checkpoint(epoch, val_mAP)
            self.train_metric.reset()
            self.pseudo_metric.reset()
            self.val_metric.reset()
        return self.best_val_mAP

    def early_stopping(self, pseuso_mAP: float, epoch: int) -> bool:
        if not self.cfg.train.get("early_stopping", False):
            return False
        epochs_wait = self.cfg.train.early_stopping * self.pseudo_update_frequency
        if epoch == 0:
            self.pseudo_mAP_start = pseuso_mAP
            return False
        elif epoch > epochs_wait and pseuso_mAP < self.pseudo_mAP_start:
            return True
        return False

    def train(self, epoch: int) -> float:
        self.model.train()
        for (imgs, (filename, pseudo_labels, target_labels)) in tqdm(self.train_dataloader, desc=f"[epoch: {epoch+1}] Training batch"):
            imgs = imgs.to(self.device)
            pseudo_labels = pseudo_labels.to(self.device)
            target_labels = target_labels.to(self.device)
            
            preds = F.softmax(self.model(imgs), dim=-1)
            self.train_metric.update(preds, target_labels)
            self.pseudo_metric.update(pseudo_labels, target_labels)
            kl_loss = self.loss(preds, pseudo_labels)
            
            # train the network
            self.optimizer.zero_grad()
            kl_loss.backward()
            self.optimizer.step()
            
            # update latent params of psuedo labels (Sec. 3.2 eq. 7)
            if (epoch + 1) > self.warmup and (epoch + 1) % self.pseudo_update_frequency == 0:
                pseudo_labels.requires_grad = True
                pseudo_labels_loss = self.loss(pseudo_labels, preds)
                pseudo_labels_grad = torch.autograd.grad(pseudo_labels_loss, pseudo_labels)[0]
                
                psi_yu = torch.exp(-0.5 * ((pseudo_labels - 0.5) / self.sigma)**2) / (self.sigma * math.sqrt(2 * math.pi))
                latent_pseudo_labels = torch.log(pseudo_labels / (1 - pseudo_labels)) # sigmoid inverse
                
                # eq. 7
                latent_pseudo_labels -= psi_yu * pseudo_labels_grad
                
                # sigmoid; detach before saving to cache, else dataloader will throw error
                pseudo_labels = torch.sigmoid(latent_pseudo_labels).clone().detach()
                # update the pseudo labels in the dataset
                self.update_pseudo_labels(filename, pseudo_labels)
        
        total_loss = self.loss.compute()
        AP_per_class = self.train_metric.compute()
        mAP = torch.mean(torch.tensor(list(AP_per_class.values())))
        pseudo_mAP = torch.mean(torch.tensor(list(self.pseudo_metric.compute().values())))
        return total_loss.item(), mAP.item(), pseudo_mAP.item()
    
    @torch.inference_mode()
    def validation(self, epoch: int) -> float:
        """calculate validation mAP"""
        self.model.eval()
        for imgs, targets in tqdm(self.val_dataloader, desc=f"[epoch: {epoch+1}] Validation batch"):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            preds = F.softmax(self.model(imgs), dim=-1)
            self.val_metric.update(preds, targets)

        AP_per_class = self.val_metric.compute()
        mAP = torch.mean(torch.tensor(list(AP_per_class.values())))
        return mAP.item()

    def update_pseudo_labels(self, filename: Tuple[str], pseudo_labels: torch.Tensor):
        for i in range(len(filename)):
            tensor_location = os.path.join(self.cfg.clip_cache.pseudo_cache_dir, filename[i].split('.')[0] + '.pt')
            torch.save(pseudo_labels[i], tensor_location)
    
    def resume_checkpoint(self) -> int:
        """Resume from a saved checkpoint

        Returns:
            checkpoint epoch (int): The epoch to start resuming training from
        """
        start_epoch = 0
        if self.cfg.train.resume.ckpt_path is not None:
            assert os.path.isfile(
                self.cfg.train.resume.ckpt_path), "Error: no checkpoint found!"
            assert len(os.listdir(
                self.cfg.clip_cache.pseudo_cache_dir)), "Cache directory is empty!"
            log.info('resuming from checkpoint')
            self.cfg.paths.log_dir = os.path.dirname(self.cfg.train.resume.ckpt_path) # set output directory same as resume directory
            checkpoint = torch.load(self.cfg.train.resume.ckpt_path)
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict, strict=True)
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.loss.load_state_dict(checkpoint['train_loss_state_dict'])
            self.best_val_mAP = checkpoint['best_val_mAP']
            start_epoch = checkpoint['epoch']
        else:
            # clear old cache if any and create new cache directory
            try:
                shutil.rmtree(self.cfg.clip_cache.pseudo_cache_dir)
                log.warning('cleared old cache')
            except Exception as e:
                log.info('no old pseudo label cache found')
            log.info(f"Creating pseudo label cache dir: {self.cfg.clip_cache.pseudo_cache_dir}")
            os.makedirs(self.cfg.clip_cache.pseudo_cache_dir, exist_ok=True)
        return start_epoch
    
    def save_checkpoint(self, epoch: int, val_mAP: float):
        """training checkpointing"""
        if not self.cfg.train.get("enable_checkpointing", False):
            self.best_val_mAP = max(self.best_val_mAP, val_mAP)
            return
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'train_loss_state_dict': self.loss.state_dict(),
            'best_val_mAP': self.best_val_mAP
        }
        # f'epoch_{epoch+1}_val_mAP_{val_mAP}.ckpt'
        torch.save(save_dict, os.path.join(self.cfg.paths.output_dir, 'last.ckpt'))
        log.info(f'[epoch: {epoch+1}] saved checkpoint')
        if val_mAP > self.best_val_mAP:
            self.best_val_mAP = val_mAP
            torch.save(save_dict, os.path.join(self.cfg.paths.output_dir, 'best.ckpt'))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    print_config_tree(cfg)
    
    # initialise seeds here!
    set_seed(cfg.train.seed)
    
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    group = None
    if cfg.logger.get("group", None):
        group = cfg.logger.group + "_" + cfg.task_name
    wandb.init(
        entity=cfg.logger.entity,
        group=group,
        project=cfg.logger.project,
        name=f"{cfg.task_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}",
        config=wandb_config,
        id=cfg.logger.id,  # replace with run id when resuming a run
        resume=cfg.logger.resume,
        tags=cfg.logger.tags,
        allow_val_change=cfg.logger.allow_val_change,
        settings=wandb.Settings(start_method="thread"),
        reinit=True,
    )
    
    trainer = Trainer(cfg)
    best_val_mAP = trainer.fit()
    log.info(f'best_val_mAP: {best_val_mAP}')
    return best_val_mAP


if __name__=='__main__':
    main()