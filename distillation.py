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

    print("Instantiating model", flush=True)
    # model = ProposedRanker(lr=0.00003, warmup_steps=1000, cache_dir=f"./cache/colbert_{trainer.precision}/")
    model = ProposedRanker(lr=1e-5, warmup_steps=1000, cache_dir=f"./cache/colbert_{trainer.precision}/")
    data_processor = ProposedDataProcessor(query_limit=10000, cache_dir=f"./cache/graphs_{trainer.precision}/")

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
