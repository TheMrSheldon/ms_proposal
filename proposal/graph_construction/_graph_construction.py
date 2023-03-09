from torch_geometric.data import Data

from typing import Optional


class GraphConstruction:
    def __call__(self, input: str, device: Optional[str] = None) -> Data:
        raise NotImplementedError
