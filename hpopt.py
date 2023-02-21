#! /usr/bin/env python3

from pathlib import Path
import hydra
from hydra.utils import instantiate as hydra_inst
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything, LightningDataModule

import common
from optuna import create_study, TrialPruned
from optuna.pruners import MedianPruner
from optuna.trial import Trial
from proposal import ProposedDataProcessor, ProposedRanker

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import warnings


# https://github.com/optuna/optuna-examples/issues/166#issuecomment-1403112861
class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise TrialPruned(message)


def _objective(trial: Trial, config: DictConfig) -> float:
    # Declare hyperparameters
    lr = trial.suggest_float("lr", 1e-7, 1e-3)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 3000)
    sparsity_tgt = trial.suggest_float("sparsity_tgt", 0, 10)
    alpha = trial.suggest_float("alpha", 0, 1)
    topk = trial.suggest_float("topk", 0, 1)

    print("Running new test using:")
    print(f"\tlr: {lr}")
    print(f"\twarmup_steps: {warmup_steps}")
    print(f"\tsparsity_tgt: {sparsity_tgt}")
    print(f"\talpha: {alpha}")
    print(f"\ttopk: {topk}")

    # Init Model, Trainer, and Dataloader
    checkpointcb = ModelCheckpoint(dirpath=config.checkpoint_path, save_top_k=-1, filename="{epoch:02d}")
    callbacks = [PyTorchLightningPruningCallback(trial, monitor="val_loss"), checkpointcb]
    trainer = hydra_inst(config.trainer, callbacks=callbacks, enable_progress_bar=False)
    assert isinstance(trainer, Trainer)

    model = ProposedRanker(
        lr=lr,
        warmup_steps=warmup_steps,
        sparsity_tgt=sparsity_tgt,
        alpha=alpha,
        topk=topk,
        cache_dir=f"./cache/colbert_{trainer.precision}/",
    )
    data_processor = ProposedDataProcessor(query_limit=10000, cache_dir=f"./cache/graphs_{trainer.precision}/")

    datamodule = hydra_inst(config.datamodule, data_processor=data_processor)
    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)

    trainer.fit(model, datamodule=datamodule)
    print(trainer.callback_metrics["val_loss"].item())
    return trainer.callback_metrics["val_loss"].item()


@hydra.main(config_path="hydra_conf", config_name="optuna", version_base=None)
def main(config: DictConfig):
    seed_everything(config.seed)
    common.set_cuda_devices_env(config.used_gpus)

    pruner = MedianPruner()

    storage_file = Path("optuna") / f"{config.run_name}.db"
    storage_file.parent.mkdir(exist_ok=True)
    storage = f"sqlite:///{storage_file.absolute()}"
    study = create_study(
        study_name="ranker",
        storage=storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: Trial) -> float:
        return _objective(trial, config)

    study.optimize(objective, n_trials=100, timeout=None)

    print(f"Number of finished trials: {len(study.trials)}")

    trial = study.best_trial
    print("Best trial:")
    print(f"\tValue: {trial.value}")
    print("\tParams:")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")


if __name__ == "__main__":
    main()
