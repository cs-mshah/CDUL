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
    save_root = cfg.paths.get("save_root", None)
    
    dataset = hydra.utils.instantiate(cfg.data.dataset, transform=preprocess, target_transform=target_transform)
    
    clip_cache = CLIPCache(dataset, object_categories, save_root, thresh=cfg.clip_cache.get("thresh", 0.5), temperature=temperature, snippet_size=cfg.clip_cache.get("batch_size", 3), batch_size=cfg.clip_cache.get("batch_size", 8))
    
    # Section 3.1.1: save soft global similarity vectors
    log.info("Saving global similarity vectors...")
    clip_cache.save(mode='global')
    
    dataset = hydra.utils.instantiate(cfg.data.dataset, target_transform=target_transform)
    
    # create a new dataset with tile crops
    tile_crop_dataset = TileCropDataset(dataset)
    
    # Section 3.1.2, 3.1.3: save soft aggregation similarity vectors
    clip_cache = CLIPCache(tile_crop_dataset, object_categories, save_root, thresh=cfg.clip_cache.get("thresh", 0.5), temperature=temperature, snippet_size=cfg.clip_cache.get("batch_size", 3), batch_size=cfg.get("cache_batch_size", 8))
    
    log.info("Saving soft aggregation similarity vectors...")
    clip_cache.save(mode='aggregate')

if __name__=='__main__':
    main()