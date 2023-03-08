#! /usr/bin/env python3

from pathlib import Path

import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint

import common
from proposal import ProposedDataProcessor, ProposedRanker


@hydra.main(config_path="hydra_conf", config_name="distillation", version_base=None)
def main(config: DictConfig):
    print("Setting distillation up", flush=True)
    seed_everything(config.seed)
    common.set_cuda_devices_env(config.used_gpus)

    print("Instantiating trainer", flush=True)
    checkpointcb = ModelCheckpoint(dirpath=config.checkpoint_path, save_top_k=-1, filename="{epoch:02d}")
    trainer = hydra_inst(config.trainer, callbacks=[checkpointcb])
    assert isinstance(trainer, Trainer)

    print("Fetching cache paths", flush=True)
    keys = {"precision": trainer.precision}
    cache_root = Path(config.cache_root)
    model_cache = cache_root/config.model_cache.format(**keys) if config.model_cache else None
    processor_cache = cache_root/config.processor_cache.format(**keys) if config.processor_cache else None
    print(f"Model cache: {model_cache}")
    print(f"Processor cache: {processor_cache}")

    print("Instantiating model", flush=True)
    lr = 3e-5
    warmup_steps = 3000
    sparsity_tgt = 3
    alpha = 0.5
    topk = 0.6

    print("Running new test using:")
    print(f"\tlr: {lr}")
    print(f"\twarmup_steps: {warmup_steps}")
    print(f"\tsparsity_tgt: {sparsity_tgt}")
    print(f"\talpha: {alpha}")
    print(f"\ttopk: {topk}")  
    model = ProposedRanker(lr=lr, warmup_steps=warmup_steps, alpha=alpha, sparsity_tgt=sparsity_tgt, topk=topk, cache_dir=model_cache)
    data_processor = ProposedDataProcessor(query_limit=10000, cache_dir=processor_cache)

    print("Instantiating datamodule", flush=True)
    datamodule = hydra_inst(config.datamodule, data_processor=data_processor)
    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)

    checkpoint = None
    if config.checkpoint is not None:
        checkpoint = Path(config.checkpoint_path) / config.checkpoint
        print(f"Resuming from checkpoint at {checkpoint}")
    print("Training", flush=True)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
