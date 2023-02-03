import torch
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import unbatch
from transformers import DistilBertTokenizer


class DocEncoder(LightningModule):
    def __init__(self, feature_size: int, hidden_size: int, steps: int = 1) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.steps = steps
        self.gcn1 = GCNConv(in_channels=feature_size, out_channels=hidden_size)
        self.gcn2 = GCNConv(in_channels=hidden_size, out_channels=feature_size)
        self.gat_alpha = GATConv(in_channels=feature_size, out_channels=feature_size)
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def _message_passing(self, x, edge_index, edge_mask) -> torch.Tensor:
        x = self.gcn1(x, edge_index, edge_mask).relu()
        x = self.gcn2(x, edge_index, edge_mask).relu()
        return x

    def _update_graph_structure(self, x, edge_index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_mask = torch.ones((edge_index.shape[1], 1))
        for _ in range(self.steps):
            # For now, we use GAT alpha. In future, we will use some distribution.
            _, (_, edge_mask) = self.gat_alpha(x, edge_index, return_attention_weights=True)
            x = self._message_passing(x, edge_index, edge_mask)
        return x, edge_index, edge_mask

    def _compute_graph_embedding(self, x, batch) -> torch.Tensor:
        # We can't do pooling here since ColBERT's late interaction will require (batch, words, feature_size) tensors
        # but pooling would result in a (batch, feature_size) tensor
        # return global_mean_pool(x, batch)  # (batch, feature_size)
        return pad_sequence(unbatch(x, batch), batch_first=True)  # (batch, words, feature_size)

    def forward(self, x, edge_index, batch, **_) -> tuple[torch.FloatTensor, Data]:
        x, _, edge_mask = self._update_graph_structure(x, edge_index)
        if not self.training:  # on inference we want to hard mask the document
            edge_mask = edge_mask.ge(0.5).float()
        emb = self._compute_graph_embedding(x, batch)
        return emb, Data(x=x, edge_index=edge_index, edge_weight=edge_mask, batch=batch)
