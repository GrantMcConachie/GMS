"""
modules for data loading
"""

import os

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset


class GraphDataset(InMemoryDataset):
    """
    Dataset definition for the mixture dataset
    """
    def __init__(
            self,
            data_dir,
            transform=None,
            pre_transform=None
    ):
        super().__init__(None, transform, pre_transform)
        self.data_dir = data_dir

    def len(self):
        return len(os.listdir(self.data_dir))

    def get(self, idx):
        graph = os.listdir(self.data_dir)[idx]
        return torch.load(os.path.join(self.data_dir, graph))


if __name__ == '__main__':
    data_dir = ".data/processed/train"
    dataset = GraphDataset(data_dir)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        follow_batch=['mix_1_x', 'mix_2_x']
    )
    for i in dataloader:
        print("\n")
        print(i)
