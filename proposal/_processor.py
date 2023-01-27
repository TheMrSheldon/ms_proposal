from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Iterable, NamedTuple

import torch
from ranking_utils.model.data import DataProcessor
from torch_geometric.data import Batch as tgBatch
from torch_geometric.data import Data

# from torch_geometric.utils.convert import to_scipy_sparse_matrix
from transformers import DistilBertModel, DistilBertTokenizer


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
        # We can't use python's hash here since it is not consistent across runs
        # key = hash(doc).to_bytes(8, "big", signed=True).hex()
        key = sha1(doc.encode(), usedforsecurity=False).hexdigest()
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
        return Input(doc=doc, query=query[: self.query_limit])

    def get_model_batch(self, inputs: Iterable[Input]) -> Batch:
        docs, queries = zip(*inputs)
        doc_in = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        query_in = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        return Batch(
            docs={"input_ids": doc_in["input_ids"], "attention_mask": doc_in["attention_mask"]},
            queries={"input_ids": query_in["input_ids"], "attention_mask": query_in["attention_mask"]},
            doc_graphs=self._construct_doc_batch([input.doc for input in inputs]),
        )
