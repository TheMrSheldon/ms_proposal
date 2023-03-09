from torch_geometric.data import Data

import torch
from ._graph_construction import GraphConstruction
from transformers import DistilBertModel, DistilBertTokenizer

from typing import Optional


class GraphOfWord(GraphConstruction):

    def __init__(self, window_size: int = 3) -> None:
        super().__init__()
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.window_size = window_size

    def __call__(self, input: str, device: Optional[str] = None) -> Data:
        # Calculate the embedding layer of the bert model for the node features
        input_ids = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        x = self.bert.embeddings(input_ids).squeeze()  # (seq_length, dim)
        num_tokens = x.size(0)

        num_edges = 2*self.window_size*(num_tokens-self.window_size-1)+self.window_size**2+self.window_size
        edges = torch.LongTensor(size=(num_edges, 2), device=device)
        idx = 0
        for word in range(num_tokens):
            for i in range(max(0, word-self.window_size), min(num_tokens, word+self.window_size+1)):
                if i != word:
                    edges[idx][0] = word
                    edges[idx][1] = i
                    idx += 1
        assert idx == edges.size(0), f"{idx} == {edges.size(0)} not met"
        assert x.size() == (num_tokens, 768)

        return Data(x, edges.T)
