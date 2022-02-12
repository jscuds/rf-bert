from typing import Dict, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class TensorRunningAverages:
    """Averages values over an arbitrary-size logging window."""
    _store_sum: Dict[str, torch.Tensor]
    _store_total: Dict[str, torch.Tensor]

    def __init__(self):
        self._store_sum = {}
        self._store_total = {}
    
    def keys(self) -> Set[str]:
        return set(self._store_sum.keys())

    def update(self, key: str, val: Union[float, torch.Tensor]) -> None:
        if key not in self._store_sum:
            self.clear(key)
        if isinstance(val, torch.Tensor):
            self._store_sum[key] += val.detach().cpu()
        else:
            self._store_sum[key] += val

        self._store_total[key] += 1

    def get(self, key: str) -> float:
        total = max(self._store_total.get(key).item(), 1.0)
        return (self._store_sum[key] / float(total)).item() or 0.0
    
    def clear(self, key: str) -> None:
        self._store_sum[key] = torch.tensor(0.0, dtype=torch.float64)
        self._store_total[key] = torch.tensor(0, dtype=torch.int32)
    
    def clear_all(self) -> None:
        for key in self._store_sum:
            self.clear(key)

def train_test_split(dataset: Dataset, batch_size: int, shuffle: bool = True, 
                     drop_last: bool= False, train_split: float = 0.8, seed: int = 42) -> Tuple[DataLoader,DataLoader]:
    """
    Given a Dataset object, creates train_dataloader and test_dataloader objects based on `train_split` size.  
    
    Ref: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    """
    # Create data indices for train and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating data samplers and DataLoaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler, drop_last = drop_last, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size = batch_size, sampler = test_sampler, drop_last = drop_last, pin_memory=True)

    return train_loader, test_loader
