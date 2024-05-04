import os
from typing import List
from multiprocessing import Pool, set_start_method
import hydra
from omegaconf import DictConfig
import clip
import torch
import rootutils
from loguru import logger as log
import lovely_tensors as lt
from functools import partial

lt.monkey_patch()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.data import TileCropDataset, CLIPCache
from src.utils.rich_utils import print_config_tree


def get_indices(cache_dir: str, images_path: List[str], gpus: int = 1, modulo: int = 0):
    """returns indices of images_path which are not present in the cache_dir"""
    if not os.path.exists(cache_dir):
        indices = list(range(len(images_path)))
        return indices
    log.info("Continuing from existing generated cache...")
    cache_files = os.listdir(cache_dir)
    cache_files = [os.path.basename(f).split(".")[0] for f in cache_files]
    base_img = lambda img: os.path.basename(img).split(".")[0]
    indice_modulo = (
        lambda img: int(os.path.basename(img).split(".")[0].split("_")[1]) % gpus
        == modulo
    )
    indices = [
        i
        for i, img in enumerate(images_path)
        if base_img(img) not in cache_files and indice_modulo(img)
    ]
    return indices


def process_cache(i, cfg, object_categories, temperature, target_transform):
    device = torch.device("cuda", i)
    temperature = temperature.to(device)
    dataset = hydra.utils.instantiate(
        cfg.data.dataset, target_transform=target_transform
    )
    indices = get_indices(
        cfg.clip_cache.aggregate_cache_dir,
        dataset.images,
        gpus=cfg.clip_cache.get("gpus", 1),
        modulo=i,
    )
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    # create a new dataset with tile crops
    sz = (cfg.clip_cache.snippet_size, cfg.clip_cache.snippet_size)
    tile_crop_dataset = TileCropDataset(subset_dataset, tile_size=sz)
    log.info(f"Processing cache of size {len(tile_crop_dataset)} on GPU {i}")

    # Section 3.1.2, 3.1.3: save soft aggregation similarity vectors
    clip_cache = CLIPCache(
        tile_crop_dataset,
        object_categories,
        cfg.clip_cache.global_cache_dir,
        cfg.clip_cache.aggregate_cache_dir,
        thresh=cfg.clip_cache.thresh,
        temperature=temperature,
        snippet_size=cfg.clip_cache.snippet_size,
        batch_size=cfg.clip_cache.batch_size,
        num_workers=cfg.clip_cache.num_workers,
        device=device,
    )

    log.info("Saving soft aggregation similarity vectors...")
    clip_cache.save(mode="aggregate")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print_config_tree(cfg)
    object_categories = hydra.utils.instantiate(cfg.data.object_categories)
    target_transform = hydra.utils.instantiate(
        cfg.data.target_transform, transform_type="filename"
    )

    model, preprocess = clip.load(cfg.model.clip.name)
    temperature = model.logit_scale.data

    if cfg.clip_cache.mode in ["global", "all"]:
        dataset = hydra.utils.instantiate(
            cfg.data.dataset, transform=preprocess, target_transform=target_transform
        )

        clip_cache = CLIPCache(
            dataset,
            object_categories,
            cfg.clip_cache.global_cache_dir,
            cfg.clip_cache.aggregate_cache_dir,
            thresh=cfg.clip_cache.thresh,
            temperature=temperature,
            snippet_size=cfg.clip_cache.snippet_size,
            batch_size=cfg.clip_cache.batch_size,
            num_workers=cfg.clip_cache.num_workers,
        )

        log.info("Saving global similarity vectors...")
        clip_cache.save(mode="global")

    if cfg.clip_cache.mode in ["aggregate", "all"]:
        del model, preprocess
        torch.cuda.empty_cache()

        set_start_method("spawn")

        num_gpus = cfg.clip_cache.get("gpus", 1)
        with Pool(num_gpus) as p:
            partial_process_cache = partial(
                process_cache,
                cfg=cfg,
                object_categories=object_categories,
                temperature=temperature,
                target_transform=target_transform,
            )
            p.map(partial_process_cache, range(num_gpus))


if __name__ == "__main__":
    main()
