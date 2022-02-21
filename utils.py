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

def log_wandb_histogram(list_of_values: list, description: str, step: int, epoch: int):
    """Logs a histogram of `list_of_values` to W&B."""
    lower_description = '_'.join(description.lower().split())
    data = [[val] for val in list_of_values]
    table = wandb.Table(data=data, columns=[lower_description])
    histogram = wandb.plot.histogram(
        table, lower_description,
        title=f"{description} Histogram, Epoch {epoch+1}"
    )
    wandb.log({
            lower_description: histogram, 
            'epoch': epoch,
            'step': step
    })



def elmo_sentence_decode(sent: torch.Tensor) -> str:
    """
    Decodes a sentence of ELMo character_ids with the shape [1,sequence_length,50] into a readable sentence string.
    Useful for debugging when you only have a ParaDatasetElmo object and its various tensor/tuple/sent_id returns.
    """
    sent = tuple([tuple(word.tolist()) for word in sent.squeeze()])
    full_sent = []
    for word in sent:
        if word == tuple([0]*50): continue
        word_concat = []
        for char in word:
            # print(char)
            if char == 259: continue
            if char == 260: break
            if char == 261: break
            word_concat.append(chr(char-1))
        full_sent.append(''.join(word_concat))
    return ' '.join(full_sent)

def elmo_word_decode(word: Tuple) -> str:
    """
    Decodes a word tuple of ELMo character_ids with length=50 into a readable sentence string.
    Useful for debugging when you only have a ParaDatasetElmo object and its various tensor/tuple/sent_id returns.
    """
    word_concat = []
    for char in word:
        if char == 259: continue
        if char == 260: break
        word_concat.append(chr(char-1))

    return ''.join(word_concat)