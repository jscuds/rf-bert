from typing import Callable, Tuple

import datasets
import numpy as np
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

def load_rotten_tomatoes(tokenizer: Callable) -> Tuple[Dataset, Dataset]:
    dataset = datasets.load_dataset('rotten_tomatoes')
    def tokenize_ex(e):
        e['tokenized_text'] = tokenizer(e['text'])
        return e
    dataset = dataset.map(tokenize_ex, batched=False)
    dataset.set_format(type='torch', columns=['tokenized_text', 'label'])
    return dataset['train'], dataset['test'] # rt also has a 'validation' set