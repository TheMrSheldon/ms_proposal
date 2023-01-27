from pathlib import Path
from typing import Any, Union

import torch
import torch.nn.functional as F
from ranking_utils.model import Ranker
from torch.linalg import vector_norm
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch_geometric.data import Data

# from torch_geometric.utils.convert import to_scipy_sparse_matrix
from transformers import get_constant_schedule_with_warmup

from common.utils import tensor_hash, unbatch_edge_attr

from ._colbert import ColBERT
from ._doc_encoder import DocEncoder
from ._processor import Batch


class ProposedRanker(Ranker):
    def __init__(self, lr: float, warmup_steps: int, cache_dir: str = "./cache/colbert/") -> None:
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.cache_dir = Path(cache_dir)
        self.doc_encoder = DocEncoder(feature_size=768, hidden_size=768)
        self.colbert: ColBERT = ColBERT.from_pretrained(
            "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        )
        self.mseloss = MSELoss()
        self.bce = BCEWithLogitsLoss(reduction="none")

        # Freeze colbers parameters since we only want to train the doc_encoder for now
        for p in self.colbert.parameters():
            p.requires_grad = False

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _fw_colbert_single(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # We only use unmasked entries of the input_ids to compute a hash for the input. Otherwise we may run into the
        # following issue:
        # Let tokenized inputs be of different dimensions i.e. the following input_ids:
        # id1: [ 101, 7592, 2088,  102]    and    id2: [ 101, 3231,  102]
        # When batched id2 will be extended to [ 101, 3231,  102, 0] to match the dimensions of id1 but these 0's are
        # masked in the attention masks:
        # am1: [ 1, 1, 1, 1]    and    am2: [ 1, 1, 1, 0]
        # In another batch, however, the same input as for id2 would have input_ids: [ 101, 3231,  102, 0, 0] with the
        # last two entries masked out and a simple hash of the input_ids would not recognize it to be effectively the
        # same input.
        input_length = torch.nonzero(attention_mask)[-1] + 1
        unmasked = input_ids[:input_length]
        key = tensor_hash(unmasked)
        cache_file = self.cache_dir / f"{key}"
        if cache_file.exists():
            data = torch.load(cache_file, map_location=self.device)
            assert isinstance(data, torch.Tensor)
        else:
            input = {"input_ids": unmasked.unsqueeze(0), "attention_mask": attention_mask[:input_length].unsqueeze(0)}
            data = self.colbert.forward_representation(input).squeeze()
            torch.save(data, cache_file)
        data = F.pad(input=data, pad=(0, 0, 0, attention_mask.size(0) - input_length), mode="constant", value=0)
        assert data.size(0) == input_ids.size(0)
        return data

    @torch.no_grad()
    def _forward_colbert_representation_or_load_from_cache(self, input: dict[str, torch.LongTensor]) -> torch.Tensor:
        out = torch.stack([self._fw_colbert_single(*pair) for pair in zip(input["input_ids"], input["attention_mask"])])
        return out

    def _sparsity(self, doc_graphs: Data) -> torch.Tensor:
        # Calculate the l2-norm for each graph's edge_weights
        edge_attrs = unbatch_edge_attr(doc_graphs.edge_weight, doc_graphs.edge_index, doc_graphs.batch)
        norms = [vector_norm(v, ord=2) for v in edge_attrs]
        return torch.tensor(norms, device=self.device)

    def forward(
        self, batch: Batch, return_all: bool = False
    ) -> Union[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor, Data]]:
        docs_graphs, docs, queries = batch.doc_graphs, batch.docs, batch.queries
        query_vecs = self._forward_colbert_representation_or_load_from_cache(queries)
        doc_vecs, doc_graphs = self.doc_encoder(**docs_graphs.to_dict())
        classlbl = self.colbert.forward_aggregation(
            query_vecs, doc_vecs, queries["attention_mask"], docs["attention_mask"]
        )
        if return_all:
            return classlbl, (query_vecs, doc_vecs, doc_graphs)
        return classlbl

    def training_step(self, batch) -> torch.Tensor:
        model_batch, labels, _ = batch
        assert isinstance(model_batch, Batch)
        pred, (query_vecs, doc_vecs, doc_graphs) = self(model_batch, return_all=True)
        assert isinstance(doc_graphs, Data)
        teacher_doc_vecs = self._forward_colbert_representation_or_load_from_cache(model_batch.docs)
        # teacher_pred = self.colbert.forward_aggregation(query_vecs, teacher_doc_vecs)

        distillation_loss = self.mseloss(doc_vecs, teacher_doc_vecs)
        sparsity = self._sparsity(doc_graphs)
        classification_loss = self.bce(pred.flatten(), labels.flatten())

        # Ablation: different combinations of these lossfunctions and their effect on training
        loss = torch.mean(distillation_loss + sparsity + classification_loss)
        # loss = torch.mean(distillation_loss + classification_loss)
        self.log_dict(
            {
                "loss/total": loss,
                "loss/distillation": torch.mean(distillation_loss),
                "loss/sparsity": torch.mean(sparsity),
                "loss/classification": torch.mean(classification_loss),
            }
        )
        return loss

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
