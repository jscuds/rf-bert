from typing import Callable, List, Tuple

import collections
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


def prepare_dataset_with_elmo_tokenizer(dataset, text_columns: List[str], max_length: int):
    """Tokenizes 'text' field and returns dataset with each column of `text_columns` transformed to token IDs,
        as well as a 'label' column.
    """
    elmo_tokenizer = MosesTokenizer('en', no_escape=True)
    def text_to_ids(e):
        for col in text_columns:
            text = elmo_tokenizer(e[col])
            # TODO: Figure out how to do this successfully in batch.
            text_ids = batch_to_ids([text])[0]
            # Pad to max length.
            text_ids = text_ids[:max_length]
            pad_size = max_length - len(text_ids)
            if pad_size > 0:
                padding = torch.zeros(pad_size, 50, dtype=int)
                text_ids = torch.cat((text_ids, padding), dim=0)
            e[col] = text_ids
        return e

    # Caching sometimes causes weird issues with .map(), if you see that then
    # add the load_from_cache_file=False argument below.
    dataset = dataset.map(text_to_ids, batched=False, load_from_cache_file=False)
    return dataset


def dataloader_from_dataset(
        dataset: datasets.Dataset,
        text_columns: List[str], label_columns: List[str],
        batch_size: int, shuffle: bool, drop_last: bool
    ) -> DataLoader:
    """Creates a dataloader from `dataset` with `text_ids` and `labels` keys. Makes sure things
    are collated properly and returned as a batched (...text_columns, ...label_columns) tuple.
    """
    def collate_fn(batch):
        """`batch` is a list with `batch_size` elements. Each element is a dict with
        `label` and `text_ids` keys."""
        d = collections.defaultdict(list)
        for el in batch:
            for k in text_columns:
                d[k].append(torch.stack(el[k])) # shape (seq_length, 50)
            for k in label_columns:
                d[k].append(el[k]) # shape ()
        # stack all lists into tensors and return in the proper order
        return tuple(torch.stack(d[k]) for k in text_columns) + tuple(torch.stack(d[k]).float() for k in label_columns)
    return DataLoader(dataset,
        batch_size=batch_size, collate_fn=collate_fn,
        shuffle=True, drop_last=drop_last, pin_memory=True
    )


def load_rotten_tomatoes(batch_size: int, max_length: int, drop_last: bool = True, num_examples: int = None) -> Tuple[DataLoader, DataLoader]:
    """Returns tuple of (train_dataloader, test_dataloader)."""
    dataset = datasets.load_dataset('rotten_tomatoes').shuffle(seed=42)
    if num_examples:
        # For some reason, truncating datasets like this returns a plain dict,
        # so need to call Dataset.from_dict to restore as a Dataset type.
        for split in dataset:
            dataset[split] = datasets.Dataset.from_dict(dataset[split][:num_examples])

    dataset = prepare_dataset_with_elmo_tokenizer(dataset=dataset, text_columns=['text'], max_length=max_length)
    # TODO: support arbitrary tokenizer (for any model)
    train_dataset, test_dataset = dataset['train'], dataset['test'] # rt also has a 'validation' set
    
    print('Loading rotten tomatoes dataset with', num_examples, 'examples')

    train_dataset.set_format(type='torch', columns=['text', 'label'])
    test_dataset.set_format(type='torch', columns=['text', 'label'])

    train_dataloader = dataloader_from_dataset(
        train_dataset, text_columns=['text'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    test_dataloader = dataloader_from_dataset(
        train_dataset, text_columns=['text'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    return train_dataloader, test_dataloader


def load_qqp(batch_size: int, max_length: int, drop_last: bool = True, num_examples: int = None) -> Tuple[DataLoader, DataLoader]:
    dataset = datasets.load_dataset('glue', 'qqp').shuffle(seed=42)
    if num_examples:
        # For some reason, truncating datasets like this returns a plain dict,
        # so need to call Dataset.from_dict to restore as a Dataset type.
        for split in dataset:
            dataset[split] = datasets.Dataset.from_dict(dataset[split][:num_examples])

    dataset = prepare_dataset_with_elmo_tokenizer(dataset=dataset, text_columns=['question1', 'question2'], max_length=max_length)
    train_dataset, test_dataset = dataset['train'], dataset['validation'] # 'test' set has no labels, so it's not useful for us

    print('Loading rotten tomatoes dataset with', num_examples, 'examples')

    train_dataset.set_format(type='torch', columns=['question1', 'question2', 'label'])
    test_dataset.set_format(type='torch', columns=['question1', 'question2', 'label'])

    train_dataloader = dataloader_from_dataset(
        dataset=train_dataset, text_columns=['question1', 'question2'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    test_dataloader = dataloader_from_dataset(
        dataset=test_dataset, text_columns=['question1', 'question2'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    return train_dataloader, test_dataloader
