import argparse
import copy
import logging
import random
import time

import numpy as np
import torch
import tqdm
import wandb

from torch.utils.data import DataLoader

from experiment import FinetuneExperiment, RetrofitExperiment

logger = logging.getLogger(__name__)

def set_random_seed(r):
    random.seed(r)
    np.random.seed(r)
    torch.manual_seed(r)
    torch.cuda.manual_seed(r)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('experiment', type=str, 
        choices=('retrofit','finetune'))

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5,
        help='number of training epochs')
    parser.add_argument('--logs_per_epoch', type=int, default=5,
        help='log metrics this number of times per epoch')
    parser.add_argument('--num_examples', type=int, default=20_000,
        help='number of training examples')
    parser.add_argument('--max_length', type=int, default=40,
        help='max length of each sequence for truncation')
    parser.add_argument('--train_test_split', type=float, default=0.8,
        help='percent of data to use for train, in (0, 1]')

    parser.add_argument('--model_name_or_path', default='elmo', 
        choices=['elmo'], help='name of model to use')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--drop_last', type=bool, default=False,
        help='whether to drop remainder of last batch')

    # TODO add dataset so we can switch between 'quora', 'mrpc'...
    # TODO add _task_dataset so we can switch between tasks for evaluation/attack
    # TODO: additional args? dropout, 

    args = parser.parse_args()
    set_random_seed(args.random_seed)
    
    return args

def main():

    args = parse_args()

    # dictionary that matches experiment argument to its respective class
    experiment_cls = {
        'retrofit': RetrofitExperiment,
        'finetune': FinetuneExperiment
    }[args.experiment]
    experiment = experiment_cls(args)

    #########################################################
    ################## DATASET & DATALOADER #################
    #########################################################

    # js NOTE: HAD TO COPY QUORA BECAUSE CHANGING THE SPLIT CHANGED THE DATALOADERS.

    # TODO(jxm): refactor this .split() thing, it's not ideal
    train_dataset = copy.copy(experiment.dataset)
    test_dataset = copy.copy(experiment.dataset)

    train_dataset.split('train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, pin_memory=True)
    test_dataset.split('test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, pin_memory=True)

    logger.info('\n***CHECK DATASET LENGTHS:***')
    logger.info(f'len(train_dataloader.dataset) = {len(train_dataloader.dataset)}')
    logger.info(f'len(test_dataloader.dataset) = {len(test_dataloader.dataset)}')


    #########################################################
    ###################### TRAINING LOOP ####################
    #########################################################


    # WandB init and config (based on argument dictionaries in imports/globals cell)
    day = time.strftime(f'%Y-%m-%d')
    wandb.init(
        name=f'{args.experiment}_{args.model_name_or_path}_{day}', # TODO: fix/restore TITLE
        project='rf-bert',
        entity='jscuds',
        notes=None,
        config=vars(args)
    )

    # train on gpu if availble, set `device` as global variable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment.model.to(device)
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=args.learning_rate)

    # use wandb.watch() to track gradients
    # watch_log_freq is setup to log 10 times per batch: num_examples//batch_size//10
    #    which means it will log gradients every `watch_log_freq` batches
    # --> is this right? It logs grads 10x *per batch*? seems like quite often. -jxm
    watch_log_freq = args.num_examples//args.batch_size//10
    wandb.watch(experiment.model, log_freq=watch_log_freq)

    log_interval = max(len(train_dataloader) // args.logs_per_epoch, 1)
    logger.info(f'Logging metrics every {log_interval} training steps')

    epoch_start_time = time.time()
    for epoch in range(args.epochs):
        logger.info(f"Starting training epoch {epoch+1}/{args.epochs}")
        for step, (batch, targets) in tqdm.tqdm(enumerate(train_dataloader), leave=False):
            batch, targets = batch.to(device), targets.to(device)
            preds = experiment.model(batch)
            train_loss = experiment.compute_loss(preds, targets, 'train')
            
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.cpu().item()

            if step % log_interval == 0:
                logger.info(f"Running evaluation at step {step} in epoch {epoch} (logs_per_epoch = {args.logs_per_epoch})")
                # Compute eval metrics every `log_interval` batches
                experiment.model.eval() # set model in eval mode for evaluation
                with torch.no_grad():
                    for batch, targets in tqdm.tqdm(test_dataloader, leave=False):
                        targets_accum.append(targets)
                        batch, targets = batch.to(device), targets.to(device)
                        preds = experiment.model(batch)
                        experiment.compute_loss_and_update_metrics(preds, targets, 'test')
                # Compute metrics, log, and reset
                metrics_dict = experiment.compute_and_reset_metrics()
                wandb.log(metrics_dict)
                # Set model back in train mode to resume training
                experiment.model.train() 
            
            ######################################
            # Log elapsed time at end of each epoch    
            epoch_end_time = time.time()
            logger.info(f"\nEpoch {epoch+1}/{args.epochs} Total EPOCH Time: {(epoch_end_time-epoch_start_time)/60:>0.2f} min")
            wandb.log({"Epoch Time": (epoch_end_time-epoch_start_time)/60})

        epoch_start_time = time.time()

if __name__ == '__main__':
    main()