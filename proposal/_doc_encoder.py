import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import Sequential, SAGEConv, global_mean_pool

from typing import Union


class DocEncoder(torch.Module):

    def __init__(self, feature_size: int, out_size: int) -> None:
        self.new_embedding = Sequential(
            SAGEConv(in_channels=feature_size, out_channels=feature_size),
            ReLU(),
        )
        self.new_structure = Sequential(
            # TODO
        )
        self.to_embedding = Sequential(
            Linear(feature_size, out_size),
        )

    def _construct_initial_graph(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement
        raise NotImplementedError

    def _update_graph_structure(self, x, edge_index) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.new_embedding(x, edge_index)
        edge_index = self.new_structure(x, edge_index)
        return (x, edge_index)

    def _compute_graph_embedding(self, x, edge_index, batch) -> torch.Tensor:
        aggr = global_mean_pool(x, batch)
        return self.to_embedding(aggr)

    def forward(self,
                batch,
                return_graphs: bool = False
                ) -> Union[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]]:
        in_graphs = self._construct_initial_graph(batch)
        out_graphs = self._update_graph_structure(*in_graphs)
        emb = self._compute_graph_embedding(*out_graphs)
        if return_graphs:
            return emb, out_graphs
        return emb
