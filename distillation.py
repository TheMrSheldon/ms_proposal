#! /usr/bin/env python3

import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from ranking_utils.model.data.h5 import H5DataModule

import common
from proposal import ProposedDataProcessor, ProposedRanker


@hydra.main(config_path="hydra_conf", config_name="distillation", version_base=None)
def main(config: DictConfig):
    seed_everything(config.seed)
    common.set_cuda_devices_env(config.used_gpus)

    model = ProposedRanker(lr=0.00003, warmup_steps=1000)
    data_processor = ProposedDataProcessor(query_limit=10000)
    datamodule = hydra_inst(config.datamodule, data_processor=data_processor)
    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, H5DataModule)

    checkpointcb = ModelCheckpoint(
        dirpath=config.checkpoint_path,
        save_top_k=-1,
        filename="{epoch:02d}"
    )
    trainer = hydra_inst(config.trainer, callbacks=[checkpointcb])

    assert isinstance(trainer, Trainer)

    checkpoint = None
    if config.checkpoint is not None:
        checkpoint = Path(config.checkpoint_path) / config.checkpoint
        print(f"Resuming from checkpoint at {checkpoint}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint)


if __name__ == '__main__':
    main()
