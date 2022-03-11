from typing import Callable, List, Tuple

import collections
import logging
import os

import datasets
import numpy as np
import torch

from allennlp.modules.elmo import batch_to_ids
from mosestokenizer import MosesTokenizer

from torch.utils.data import BatchSampler, RandomSampler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

logger = logging.getLogger(__name__)

num_cpus = len(os.sched_getaffinity(0)) # ask the OS how many cpus we have (stackoverflow.com/questions/1006289)

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

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=drop_last, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=drop_last, pin_memory=torch.cuda.is_available())

    return train_loader, test_loader

def elmo_tokenize_and_pad(
        tokenizer: MosesTokenizer, text: str, max_length: int, hidden_size: int = 50, lowercase_inputs: bool = False
    ) -> torch.Tensor:
    """Tokenizes a single sentence and truncates/pads to length `max_length`."""
    if lowercase_inputs:
        text = text.lower()

    text = tokenizer(text)
    # TODO: Figure out how to do this successfully in batch.
    text_ids = batch_to_ids([text])[0]
    # Pad to max length.
    text_ids = text_ids[:max_length]
    pad_size = max_length - len(text_ids)
    if pad_size > 0:
        padding = torch.zeros(pad_size, 50, dtype=int)
        text_ids = torch.cat((text_ids, padding), dim=0)
    return text_ids.tolist()


def prepare_dataset_with_elmo_tokenizer(dataset, text_columns: List[str], max_length: int, lowercase_inputs: bool
    ) -> datasets.Dataset:
    """Tokenizes 'text' field and returns dataset with each column of `text_columns` transformed to token IDs,
        as well as a 'label' column.
    """
    elmo_tokenizer = MosesTokenizer('en', no_escape=True)
    def text_to_ids(e):
        for col in text_columns:
            text = e[col]
            e[col] = elmo_tokenize_and_pad(elmo_tokenizer, text, max_length, lowercase_inputs)
        return e

    # Caching sometimes causes weird issues with .map(), if you see that then
    # add the load_from_cache_file=False argument below.
    return dataset.map(text_to_ids, batched=False)


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
        batch = batch[0] # RandomSampler wraps batch in extra dimension
        return (
            tuple(torch.tensor(batch[k]) for k in text_columns) 
          + tuple(torch.tensor(batch[k]).float() for k in label_columns)
        )
    
    # epochs = 5
    sampler = BatchSampler(
        RandomSampler(dataset, replacement=False),
        batch_size = batch_size, drop_last = drop_last
    )
    return DataLoader(dataset,
        collate_fn=collate_fn,
        sampler=sampler, pin_memory=torch.cuda.is_available(),
        # num_workers=min(8, num_cpus)
    )


def load_rotten_tomatoes(
        batch_size: int, max_length: int, drop_last: bool = True, num_examples: int = None, random_seed: int = 42,
        lowercase_inputs: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
    """Returns tuple of (train_dataloader, test_dataloader)."""
    dataset = datasets.load_dataset('rotten_tomatoes').shuffle(seed=random_seed)
    if num_examples:
        # For some reason, truncating datasets like this returns a plain dict,
        # so need to call Dataset.from_dict to restore as a Dataset type.
        for split in dataset:
            dataset[split] = datasets.Dataset.from_dict(dataset[split][:num_examples])

    dataset = prepare_dataset_with_elmo_tokenizer(
        dataset=dataset, 
        text_columns=['text'], 
        max_length=max_length, 
        lowercase_inputs=lowercase_inputs
    )
    # TODO: support arbitrary tokenizer (for any model)
    train_dataset, test_dataset = dataset['train'], dataset['test'] # rt also has a 'validation' set

    logger.info('Loading Rotten Tomatoes dataset with %d examples', len(train_dataset))

    train_dataloader = dataloader_from_dataset(
        train_dataset, text_columns=['text'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    test_dataloader = dataloader_from_dataset(
        test_dataset, text_columns=['text'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    return train_dataloader, test_dataloader


def load_sst2(
        batch_size: int, max_length: int, drop_last: bool = True, num_examples: int = None, random_seed: int = 42,
        lowercase_inputs: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
    dataset = datasets.load_dataset('glue', 'sst2').shuffle(seed=random_seed)
    if num_examples:
        # For some reason, truncating datasets like this returns a plain dict,
        # so need to call Dataset.from_dict to restore as a Dataset type.
        for split in dataset:
            dataset[split] = datasets.Dataset.from_dict(dataset[split][:num_examples])

    dataset = prepare_dataset_with_elmo_tokenizer(
        dataset=dataset,
        text_columns=['sentence'],
        max_length=max_length,
        lowercase_inputs=lowercase_inputs
    )
    train_dataset, test_dataset = dataset['train'], dataset['validation'] # 'test' set has no labels, so it's not useful for us

    logger.info('Loading SST-2 dataset with %d examples', len(train_dataset))

    train_dataloader = dataloader_from_dataset(
        dataset=train_dataset, text_columns=['sentence'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    test_dataloader = dataloader_from_dataset(
        dataset=test_dataset, text_columns=['sentence'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    return train_dataloader, test_dataloader


def load_qqp(
        batch_size: int, max_length: int, drop_last: bool = True, num_examples: int = None, random_seed: int = 42,
        lowercase_inputs: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
    dataset = datasets.load_dataset('glue', 'qqp').shuffle(seed=random_seed)
    if num_examples:
        # For some reason, truncating datasets like this returns a plain dict,
        # so need to call Dataset.from_dict to restore as a Dataset type.
        for split in dataset:
            dataset[split] = datasets.Dataset.from_dict(dataset[split][:num_examples])

    dataset = prepare_dataset_with_elmo_tokenizer(
        dataset=dataset,
        text_columns=['question1', 'question2'],
        max_length=max_length,
        lowercase_inputs=lowercase_inputs
    )
    train_dataset, test_dataset = dataset['train'], dataset['validation'] # 'test' set has no labels, so it's not useful for us

    logger.info('Loading QQP dataset with %d examples', len(train_dataset))

    train_dataloader = dataloader_from_dataset(
        dataset=train_dataset, text_columns=['question1', 'question2'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    test_dataloader = dataloader_from_dataset(
        dataset=test_dataset, text_columns=['question1', 'question2'], label_columns=['label'],
        batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    return train_dataloader, test_dataloader
