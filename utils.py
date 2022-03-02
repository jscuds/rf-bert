from typing import Any, Dict, Set, Tuple, Union

import functools

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

    - 259 is the beginning of word character
    - 260 is the end of word character
    - 261 is the padding character

    From [AllenNLP's ELMo Tokenizer]
    (https://github.com/allenai/allennlp/blob/b8f92f036e37015aabe0d76aabf2722597c7686f/allennlp/data/token_indexers/elmo_indexer.py#L38-L44) 
    "char ids 0-255 come from utf-8 encoding bytes."  Based on trial and error, for some reason, the values are off by 1.  
    Example: 260 should be the "padding character", but in reality it's 261.
    """
    sent = tuple([tuple(word.tolist()) for word in sent.squeeze()])
    full_sent = []
    for word in sent:
        if word == tuple([0]*50): continue
        chars_in_word = []
        for char in word:
            if char == 259: continue # beginning of word character
            if char == 260: break    # end of word character
            if char == 261: break    # padding character
            chars_in_word.append(chr(char-1))
        full_sent.append(''.join(chars_in_word))
    return ' '.join(full_sent)

def elmo_word_decode(word: Tuple) -> str:
    """
    Decodes a word tuple of ELMo character_ids with length=50 into a readable sentence string.
    Useful for debugging when you only have a ParaDatasetElmo object and its various tensor/tuple/sent_id returns.
    """
    chars_in_word = []
    for char in word:
        if char == 259: continue # beginning of word character
        if char == 260: break    # end of word character
        chars_in_word.append(chr(char-1))

    return ''.join(chars_in_word)


def colored_text(text: Any, r: int, g: int, b: int):
    """Colors text for printing to terminal using ANSI escape codes."""
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, str(text))

red_text = functools.partial(colored_text, r=255, g=0, b=0)
blue_text = functools.partial(colored_text, r=0, g=0, b=255)
green_text = functools.partial(colored_text, r=0, g=255, b=0)

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']