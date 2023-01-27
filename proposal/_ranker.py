from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from negative_cache.handlers import CacheLossHandler
from negative_cache.losses import (
    CacheClassificationLoss,
    DistributedCacheClassificationLoss,
)
from negative_cache.negative_cache import CacheManager, FixedLenFeature
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


class Representation(NamedTuple):
    vector: torch.Tensor
    attention_mask: torch.Tensor
    graph: Optional[Data] = None


class ProposedRanker(Ranker):
    def __init__(
        self, lr: float, warmup_steps: int, hparams: dict[str, Any], cache_dir: str = "./cache/colbert/"
    ) -> None:
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
        self.handler: CacheLossHandler = self._create_negative_cache(
            cache_size=hparams["cache_size"],
            top_k=hparams["top_k"],
            feature_size=hparams["input_feature_size"],
            distributed_cache_loss=hparams["distributed_cache_loss"],
            emb_size=hparams["projection_size"],
        )

    def _create_negative_cache(
        self,
        cache_size: int,
        top_k: int,
        feature_size: int = 512,
        emb_size: int = 768,
        distributed_cache_loss: bool = False,
    ) -> CacheLossHandler:
        """Set up the negative cache for training.

        Args:
            cache_size (int): Number of documents in the cache.
            top_k (int): Number of top-k documents to sample the negative from.
            feature_size (int, optional): Dimension of input features. Defaults to 512.
            emb_size (int, optional): Dimension of embedding vectors. Defaults to 768.
            distributed_cache_loss (bool, optional): Use a distributed cache. Defaults to False.

        Returns:
            CacheLossHandler: The cache loss handler.
        """
        data_keys = (
            "input_ids",
            "attention_mask",
        )
        embedding_key = "embedding"
        specs = {
            "input_ids": FixedLenFeature(shape=[feature_size], dtype=torch.int32),
            "attention_mask": FixedLenFeature(shape=[feature_size], dtype=torch.int32),
            "embedding": FixedLenFeature(shape=[emb_size], dtype=torch.float32),
        }
        cache_manager = CacheManager(specs, cache_size=cache_size)
        CacheType = DistributedCacheClassificationLoss if distributed_cache_loss else CacheClassificationLoss

        # Function instead of lambda (see https://github.com/marceljahnke/negative-cache#special-cases)
        def score_transform(score):
            return 20*score
        cache_loss = CacheType(
            embedding_key=embedding_key,
            data_keys=data_keys,
            score_transform=score_transform,
            top_k=top_k,
        )
        cache_loss._score_documents = self._late_interaction
        handler = CacheLossHandler(
            cache_manager=cache_manager,
            cache_loss=cache_loss,
            embedding_key=embedding_key,
            data_keys=data_keys,
        )
        return handler

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

    def _enc_doc(self, input: tuple[Data, torch.Tensor]) -> Representation:
        doc_graphs, docs = input
        doc_vecs, new_graphs = self.doc_encoder(**doc_graphs.to_dict())
        return Representation(vector=doc_vecs, attention_mask=docs["attention_mask"], graph=new_graphs)

    def _enc_query(self, queries: dict[str, torch.LongTensor]) -> Representation:
        query_vecs = self._forward_colbert_representation_or_load_from_cache(queries)
        return Representation(vector=query_vecs, attention_mask=queries["attention_mask"])

    def _late_interaction(self, query: Representation, doc: Representation) -> torch.Tensor:
        return self.colbert.forward_aggregation(query.vector, doc.vector, query.attention_mask, doc.attention_mask)

    def forward(
        self, batch: Batch, return_all: bool = False
    ) -> Union[torch.FloatTensor, tuple[Representation, Representation]]:
        query_rep = self._enc_query(batch.queries)
        doc_rep = self._enc_doc((batch.doc_graphs, batch.docs))
        classlbl = self._late_interaction(query_rep, doc_rep)
        if return_all:
            return classlbl, (query_rep, doc_rep)
        return classlbl

    def training_step(self, batch) -> torch.Tensor:
        model_batch, labels, _ = batch
        assert isinstance(model_batch, Batch)
        pred, (query_rep, doc_rep) = self(model_batch, return_all=True)
        assert isinstance(query_rep, Representation)
        assert isinstance(doc_rep, Representation)
        teacher_doc_vecs = self._forward_colbert_representation_or_load_from_cache(model_batch.docs)
        # teacher_pred = self.colbert.forward_aggregation(query_vecs, teacher_doc_vecs)

        distillation_loss = self.mseloss(doc_rep.vector, teacher_doc_vecs)
        sparsity = self._sparsity(doc_rep.graph)
        # classification_loss = self.bce(pred.flatten(), labels.flatten())
        # use the negative cache for classification loss
        classification_loss = self.handler.update_cache_and_compute_loss(
            item_network=self._enc_doc,
            query_embeddings=query_rep,
            pos_item_embeddings=doc_rep,
            features=(model_batch.doc_graphs, model_batch.docs),
            writer=(self.logger.experiment, self.global_step),
        )

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
