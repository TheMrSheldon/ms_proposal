#! /usr/bin/env python3

from collections import defaultdict
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
from ranking_utils import write_trec_eval_file
from tqdm import tqdm

import common
from common.datasets.trec19passage import TREC2019Passage
from common.trec_eval import load_run_from_file, trec_evaluation
from proposal import ProposedDataProcessor, ProposedRanker
from proposal.graph_construction import GraphOfWord


@hydra.main(config_path="hydra_conf", config_name="trec_eval", version_base=None)
def main(config: DictConfig):
    seed_everything(config.seed)
    common.set_cuda_devices_env(config.used_gpus)

    result_path = Path(config.result_path)
    checkpoint_path = Path(config.checkpoint_path)

    trainer = hydra_inst(config.trainer)
    assert isinstance(trainer, Trainer)
    assert trainer.num_devices == 1

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

    print("Running eval using:")
    print(f"\tlr: {lr}")
    print(f"\twarmup_steps: {warmup_steps}")
    print(f"\tsparsity_tgt: {sparsity_tgt}")
    print(f"\talpha: {alpha}")
    print(f"\ttopk: {config.topk}")
    # graph_construction = FullyConnected()
    graph_construction = GraphOfWord(window_size=5)
    model = ProposedRanker(lr=lr, warmup_steps=warmup_steps, alpha=alpha, sparsity_tgt=sparsity_tgt, topk=config.topk, cache_dir=model_cache)
    data_processor = ProposedDataProcessor(query_limit=10000, graph_construction=graph_construction, cache_dir=processor_cache)

    datamodule = hydra_inst(config.datamodule, data_processor=data_processor)
    assert isinstance(model, LightningModule)
    assert isinstance(datamodule, LightningDataModule)
    assert isinstance(datamodule, TREC2019Passage)

    if not result_path.exists():
        print("Evaluating model")
        predictions = trainer.predict(
            model=model, dataloaders=datamodule, return_predictions=True, ckpt_path=checkpoint_path / config.checkpoint
        )
        ids = [(qid, did) for _, qid, did in datamodule.predict_dataset.ids()]
        result = defaultdict(dict[str, float])
        for entry in tqdm(predictions):
            for idx, score in zip(entry["indices"], entry["scores"]):
                q_id, doc_id = ids[idx]
                result[q_id][doc_id] = float(score)
        write_trec_eval_file(result_path, result, "test")
        result = dict(result)
    else:
        print(f"Loading past evaluation run from file {result_path}")
        result = load_run_from_file(result_path)

    qrels = datamodule.qrels()
    rl = datamodule.relevance_level()
    print(trec_evaluation(qrels, result, ["recip_rank", "map", "ndcg_cut.10", "ndcg_cut.20"], relevance_level=rl))


if __name__ == "__main__":
    main()
