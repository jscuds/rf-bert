from typing import Callable, List, Tuple

import datasets
import numpy as np
import torch

from allennlp.modules.elmo import batch_to_ids
from mosestokenizer import MosesTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

def train_test_split(dataset: Dataset, batch_size: int, shuffle: bool = True, 
                     drop_last: bool = False, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Given a Dataset object, creates train_dataloader and test_dataloader objects based on `train_split` size.  
    
    Ref: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    """
    # Create data indices for train and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating data samplers and DataLoaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=drop_last, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=drop_last, pin_memory=True)

    return train_loader, test_loader

def prepare_dataset_with_elmo_tokenizer(dataset, max_length: int):
    """Tokenizes 'text' field and returns dataset with 'text_ids' and 'label' columns."""
    elmo_tokenizer = MosesTokenizer('en', no_escape=True)
    def text_to_ids(e):
        text = elmo_tokenizer(e['text'])
        # TODO: Figure out how to do this successfully in batch.
        text_ids = batch_to_ids([text])[0]
        # Pad to max length.
        text_ids = text_ids[:max_length]
        pad_size = max_length - len(text_ids)
        if pad_size > 0:
            padding = torch.zeros(pad_size, 50, dtype=int)
            text_ids = torch.cat((text_ids, padding), dim=0)
        e['text_ids'] = text_ids
        return e

    # Caching sometimes causes weird issues with .map(), if you see that then
    # add the load_from_cache_file=False argument below.
    dataset = dataset.map(text_to_ids, batched=False)

    dataset.set_format(type='torch', columns=['text_ids', 'label'])
    return dataset

def dataloader_from_dataset(dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    """Creates a dataloader from `dataset` with `text_ids` and `labels` keys. Makes sure things
    are collated properly and returned as a (input_batch, label_batch) tuple.
    """
    def collate_fn(batch):
        """`batch` is a list with `batch_size` elements. Each element is a dict with
        `label` and `text_ids` keys."""
        labels = []
        text_ids = []
        for el in batch:
            labels.append(el['label']) # shape ()
            text_ids.append(torch.stack(el['text_ids'])) # shape (seq_length, 50)
        return torch.stack(text_ids), torch.stack(labels)
    return DataLoader(dataset,
        batch_size=batch_size, collate_fn=collate_fn,
        shuffle=True, drop_last=drop_last, pin_memory=True
    )


def load_rotten_tomatoes(batch_size: int, max_length: int, drop_last: bool = True, num_examples: int = None) -> Tuple[DataLoader, DataLoader]:
    """Returns tuple of (train_dataloader, test_dataloader)."""
    dataset = datasets.load_dataset('rotten_tomatoes').shuffle(seed=42)
    dataset = prepare_dataset_with_elmo_tokenizer(dataset=dataset, max_length=max_length)
    # TODO: support arbitrary tokenizer (for any model)
    # TODO: why does 'text_ids' field get converted back from a tensor to a list?
    train_dataset, test_dataset = dataset['train'], dataset['test'] # rt also has a 'validation' set

    if num_examples:
        train_dataset = train_dataset[:num_examples]
        test_dataset = test_dataset[:num_examples]
    print('Loading rotten tomatoes dataset with', num_examples, 'examples')

    # TODO: argparse for shuffle
    train_dataloader = dataloader_from_dataset(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_dataloader = dataloader_from_dataset(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    return train_dataloader, test_dataloader