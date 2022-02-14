from typing import Dict, Set, Tuple, Union

import numpy as np
import torch
import wandb


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

def log_wandb_histogram(list_of_values: list, description: str, epoch: int):
    """Logs a histogram of `list_of_values` to W&B."""
    lower_description = '_'.join(description.lower().split())
    data = [[val] for val in list_of_values]
    table = wandb.Table(data=data, columns=[lower_description])
    wandb.log({f'{lower_description}_epoch_{epoch+1}': wandb.plot.histogram(table,lower_description,
                title=f"{description} Histogram, Epoch {epoch+1}")})
