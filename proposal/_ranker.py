from collections import deque

from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from ranking_utils.model import Ranker, TrainingMode
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


class NegSampling():
    def __init__(self, bank_size: int, sample_size: int, warmup: int = 40000) -> None:
        assert bank_size >= sample_size
        self.bank_size = bank_size
        self.sample_size = sample_size
        self.warmup = warmup

        self.queue = deque(maxlen=self.bank_size)

    def register(self, batch: Representation):
        vectors = batch.vector.cpu()
        masks = batch.attention_mask.cpu()
        for vector, mask in zip(vectors, masks):
            input_length = torch.nonzero(mask)[-1] + 1
            self.queue.append((vector[:input_length].clone(), mask[:input_length].clone()))

    def sample(self, batch: Representation, device) -> Representation:
        entries = [self.queue.popleft() for _ in range(min(self.sample_size, len(self.queue)))]
        if not entries:
            return None
        return Representation(
            vector=self.collate_fn([v for v, _ in entries]).to(device),
            attention_mask=self.collate_fn([m for _, m in entries]).to(device),
        )

    def collate_fn(self, data_list: list[torch.Tensor]) -> torch.Tensor:
        max_length = max(x.size(0) for x in data_list)
        padded = [F.pad(input=x, pad=[0,0]*(data_list[0].dim()-1) + [0, max_length - x.size(0)], mode="constant", value=0) for x in data_list]
        return torch.stack(padded)


class ProposedRanker(Ranker):
    def __init__(self, lr: float, warmup_steps: int, cache_dir: str = "./cache/colbert/") -> None:
        super().__init__(training_mode=TrainingMode.CONTRASTIVE)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.cache_dir = Path(cache_dir)
        self.doc_encoder = DocEncoder(feature_size=768, hidden_size=768)
        self.colbert: ColBERT = ColBERT.from_pretrained(
            "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        )
        self.mseloss = MSELoss()
        self.bce = BCEWithLogitsLoss(reduction="none")
        self.neg_sampling = NegSampling(20, 10)

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
        pos_batch, _, _ = batch  # We expect to only get positives here
        assert isinstance(pos_batch, Batch)
        pred, (query_rep, doc_rep) = self(pos_batch, return_all=True)
        assert isinstance(query_rep, Representation)
        assert isinstance(doc_rep, Representation)

        # Compute classification of the sampled negatives for contrastive loss
        neg_emb_batch = self.neg_sampling.sample(batch, device=self.device)  # (samplesize, ...)
        self.neg_sampling.register(doc_rep)
        if neg_emb_batch is not None:
            with torch.no_grad():
                num_samples = neg_emb_batch.vector.size(0)
                num_queries = query_rep.vector.size(0)
                exp_queries = query_rep.vector.unsqueeze(1).expand(-1, num_samples, -1, -1)  # (batch, samplesize, ...)
                exp_neg_emb = neg_emb_batch.vector.unsqueeze(0).repeat_interleave(num_queries, dim=0)  # (batch, samplesize, ...)

                exp_q_mask = query_rep.attention_mask.unsqueeze(1).expand(-1, num_samples, -1)  # (batch, samplesize, ...)
                exp_neg_mask = neg_emb_batch.attention_mask.unsqueeze(0).repeat_interleave(num_queries, dim=0)  # (batch, samplesize, ...)

                pos_outputs = torch.exp(pred)  # (batch)
                new_shape = (-1, *exp_queries.shape[2:]) #(exp_queries.shape[0]*exp_queries.shape[1], *exp_queries.shape[2:])
                new_attn_shape = (-1, *exp_q_mask.shape[2:])
                # q_rep = Representation(exp_queries.reshape(new_shape), exp_q_mask.reshape(new_shape))
                q_rep = Representation(exp_queries.reshape(new_shape), exp_q_mask.reshape(new_attn_shape))
                new_shape = (-1, *exp_neg_emb.shape[2:])
                new_attn_shape = (-1, *exp_neg_mask.shape[2:])
                n_rep = Representation(exp_neg_emb.reshape(new_shape), exp_neg_mask.reshape(new_attn_shape))
            neg_outputs = torch.exp(self._late_interaction(q_rep, n_rep))  # (batch*samplesize, 1) # (batch, samplesize)
            # print(neg_outputs.size())
            neg_sum = neg_outputs.reshape((num_queries, num_samples)).sum(1)  # (batch)
            # print(neg_sum.size())
            classification_loss = torch.mean(-torch.log(pos_outputs / (pos_outputs+neg_sum)).flatten())
            # print(classification_loss)
            # print(classification_loss.size())
        else:
            pos_outputs = torch.exp(pred)  # (batch)
            classification_loss = torch.mean(-torch.log(pos_outputs / pos_outputs).flatten())

        # Compute teacher embedding for the distillation loss
        teacher_doc_vecs = self._forward_colbert_representation_or_load_from_cache(pos_batch.docs)
        distillation_loss = torch.mean(self.mseloss(doc_rep.vector, teacher_doc_vecs))

        # Compute sparsity of the document representation
        sparsity = torch.mean(self._sparsity(doc_rep.graph))

        # Ablation: different combinations of these lossfunctions and their effect on training
        loss = distillation_loss + sparsity + classification_loss
        # loss = distillation_loss + classification_loss
        self.log_dict(
            {
                "loss/total": loss,
                "loss/distillation": distillation_loss,
                "loss/sparsity": sparsity,
                "loss/classification": classification_loss,
            }
        )
        return loss

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
