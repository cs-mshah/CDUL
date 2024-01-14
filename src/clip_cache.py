import hydra
from omegaconf import DictConfig
import clip
import rootutils
from loguru import logger as log
import lovely_tensors as lt
lt.monkey_patch()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.data import TileCropDataset, CLIPCache
from src.utils.rich_utils import print_config_tree

@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    print_config_tree(cfg)
    object_categories = hydra.utils.instantiate(cfg.data.object_categories)
    target_transform = hydra.utils.instantiate(cfg.data.target_transform, transform_type='filename')
    
    model, preprocess = clip.load(cfg.model.clip.name)
    temperature = model.logit_scale.data # get the CLIP learnt temperature
    
    if cfg.clip_cache.mode in ['global', 'all']:
        dataset = hydra.utils.instantiate(cfg.data.dataset, transform=preprocess, target_transform=target_transform)
        
        clip_cache = CLIPCache(dataset, 
                            object_categories, 
                            cfg.clip_cache.global_cache_dir,
                            cfg.clip_cache.aggregate_cache_dir,
                            thresh=cfg.clip_cache.thresh, 
                            temperature=temperature, 
                            snippet_size=cfg.clip_cache.snippet_size, 
                            batch_size=cfg.clip_cache.batch_size,
                            num_workers=cfg.clip_cache.num_workers)
        
        # Section 3.1.1: save soft global similarity vectors
        log.info("Saving global similarity vectors...")
        clip_cache.save(mode='global')

    if cfg.clip_cache.mode in ['aggregate', 'all']:
        dataset = hydra.utils.instantiate(cfg.data.dataset, target_transform=target_transform)
        
        # create a new dataset with tile crops
        sz = (cfg.clip_cache.snippet_size, cfg.clip_cache.snippet_size)
        tile_crop_dataset = TileCropDataset(dataset, tile_size=sz)
        
        # Section 3.1.2, 3.1.3: save soft aggregation similarity vectors
        clip_cache = CLIPCache(tile_crop_dataset, 
                            object_categories, 
                            cfg.clip_cache.global_cache_dir,
                            cfg.clip_cache.aggregate_cache_dir,
                            thresh=cfg.clip_cache.thresh, 
                            temperature=temperature, 
                            snippet_size=cfg.clip_cache.snippet_size,
                            batch_size=cfg.clip_cache.batch_size,
                            num_workers=cfg.clip_cache.num_workers)
        
        log.info("Saving soft aggregation similarity vectors...")
        clip_cache.save(mode='aggregate')

if __name__=='__main__':
    main()