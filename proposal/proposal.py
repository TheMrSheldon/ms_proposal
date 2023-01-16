from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor

import torch
from transformers import get_constant_schedule_with_warmup
from typing import Any, Iterable, Tuple, Union

from ._doc_encoder import DocEncoder
from ._colbert import ColBERT

Input = Tuple[str, str]
Batch = Tuple[torch.LongTensor, torch.LongTensor]


class ProposedDataProcessor(DataProcessor):

    def __init__(self) -> None:
        super().__init__()

    def get_model_input(self, query: str, doc: str) -> Input:
        # TODO: implement
        raise NotImplementedError

    def get_model_batch(self, inputs: Iterable[Input]) -> Batch:
        # TODO: implement
        raise NotImplementedError


class ProposedRanker(Ranker):

    def __init__(self) -> None:
        super().__init__()
        self.doc_encoder = DocEncoder()
        self.colbert: ColBERT = ColBERT.from_pretrained(
            "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        )

    def forward(self,
                batch: Batch,
                return_graphs: bool = False
                ) -> Union[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]]:
        docs, queries = batch
        query_vecs = self.colbert.forward_representation(queries)
        enc_out = self.doc_encoder(docs, return_graphs)
        if return_graphs:
            doc_vecs, graphs = enc_out
        emb = self.colbert.forward_aggregation(query_vecs, doc_vecs)
        if return_graphs:
            return emb, graphs
        return emb

    def training_step(self, batch) -> torch.Tensor:
        emb, graphs = self(batch, return_graphs=True)

        sparsity = None # TODO
        classification_loss = None # TODO

        loss = sparsity + classification_loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Tuple[list[Any], list[Any]]:
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
