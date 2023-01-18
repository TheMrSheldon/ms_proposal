from itertools import product

from pathlib import Path

from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor

import torch
from torch.nn import MSELoss, BCEWithLogitsLoss
# from torch.linalg import vector_norm
from torch_geometric.data import Batch as tgBatch, Data
# from torch_geometric.utils import unbatch_edge_index
# from torch_geometric.utils.convert import to_scipy_sparse_matrix
from transformers import DistilBertModel, DistilBertTokenizer, get_constant_schedule_with_warmup
from typing import Any, Iterable, NamedTuple, Union

from ._doc_encoder import DocEncoder
from ._colbert import ColBERT


class Input(NamedTuple):
    doc: str
    query: str


class Batch(NamedTuple):
    docs: dict[str, torch.LongTensor]
    queries: dict[str, torch.LongTensor]
    doc_graphs: Data


class ProposedDataProcessor(DataProcessor):

    def __init__(self, query_limit: int, cache_dir: str = "./cache/graphs/") -> None:
        super().__init__()
        self.query_limit = query_limit
        self.cache_dir = Path(cache_dir)
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _construct_graph_or_load_from_cache(self, doc: str) -> Data:
        key = hash(doc).to_bytes(8, "big", signed=True).hex()
        cache_file = self.cache_dir / f"{key}"
        if cache_file.exists():
            data = torch.load(cache_file)
            assert isinstance(data, Data)
        else:
            # Calculate the embedding layer of the bert model for the node features
            input_ids = self.tokenizer(doc, padding=True, truncation=True, return_tensors="pt")["input_ids"]
            x = self.bert.embeddings(input_ids).squeeze()  # (seq_length, dim)
            num_tokens = x.size(0)

            # Create the edge_index for the fully connected graph with num_tokens nodes
            edge_index = torch.LongTensor(list(product(range(num_tokens), range(num_tokens)))).T

            assert x.size() == (num_tokens, 768)
            assert edge_index.size() == (2, num_tokens**2)

            data = Data(x=x, edge_index=edge_index)
            torch.save(data, cache_file)

        return data

    def _construct_doc_batch(self, docs: list[str]) -> Data:
        batch = tgBatch.from_data_list([self._construct_graph_or_load_from_cache(doc) for doc in docs])
        assert isinstance(batch, Data)
        return batch

    def get_model_input(self, query: str, doc: str) -> Input:
        query = query.strip() or "(empty)"
        doc = doc.strip() or "(empty)"
        return Input(doc=doc, query=query[:self.query_limit])

    def get_model_batch(self, inputs: Iterable[Input]) -> Batch:
        docs, queries = zip(*inputs)
        doc_in = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        query_in = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        return Batch(
            docs={"input_ids": doc_in["input_ids"], "attention_mask": doc_in["attention_mask"]},
            queries={"input_ids": query_in["input_ids"], "attention_mask": query_in["attention_mask"]},
            doc_graphs=self._construct_doc_batch([input.doc for input in inputs])
        )


class ProposedRanker(Ranker):

    def __init__(self, lr: float, warmup_steps: int) -> None:
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.doc_encoder = DocEncoder(feature_size=768, hidden_size=768)
        self.colbert: ColBERT = ColBERT.from_pretrained(
            "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        )
        self.mseloss = MSELoss()
        self.bce = BCEWithLogitsLoss()

        # Freeze colbers parameters since we only want to train the doc_encoder for now
        for p in self.colbert.parameters():
            p.requires_grad = False

    def _sparsity(self, doc_graphs: Data) -> torch.Tensor:
        raise NotImplementedError
        # Need another way to unbatch the edge_weights
        # norms = [vector_norm(v, ord=2) for v in unbatch_edge_index(doc_graphs.edge_weight, doc_graphs.batch)]
        # return torch.tensor(norms)

    def forward(self,
                batch: Batch,
                return_all: bool = False
                ) -> Union[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor, Data]]:
        docs_graphs, docs, queries = batch.doc_graphs, batch.docs, batch.queries
        query_vecs = self.colbert.forward_representation(queries)
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
        teacher_doc_vecs = self.colbert.forward_representation(model_batch.docs)
        # teacher_pred = self.colbert.forward_aggregation(query_vecs, teacher_doc_vecs)

        distillation_loss = self.mseloss(doc_vecs, teacher_doc_vecs)
        #sparsity = self._sparsity(doc_graphs)
        classification_loss = self.bce(pred.flatten(), labels.flatten())

        # Ablation: different combinations of these lossfunctions and their effect on training
        loss = distillation_loss + classification_loss  # TODO: add sparsity metric
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
