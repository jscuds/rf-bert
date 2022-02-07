import argparse
import copy
import logging
import random
import time

from pathlib import Path

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


def get_argparser() -> argparse.ArgumentParser:
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
    parser.add_argument('--rf_lambda', type=float, default=1,
        help='lambda - regularization constant for retrofitting loss')
    parser.add_argument('--rf_gamma', type=float, default=2,
        help='gamma - margin constant for retrofitting loss')
    parser.add_argument('--drop_last', type=bool, default=False,
        help='whether to drop remainder of last batch')

    # TODO add dataset so we can switch between 'quora', 'mrpc'...
    # TODO add _task_dataset so we can switch between tasks for evaluation/attack
    # TODO: additional args? dropout...
    
    return parser

def run_training_loop(args: argparse.Namespace):
    # dictionary that matches experiment argument to its respective class
    experiment_cls = {
        'retrofit': RetrofitExperiment,
        'finetune': FinetuneExperiment
    }[args.experiment]
    experiment = experiment_cls(args)

    #########################################################
    ################## DATASET & DATALOADER #################
    #########################################################

    # NOTE(js): HAD TO COPY QUORA BECAUSE CHANGING THE SPLIT CHANGED THE DATALOADERS.

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


    day = time.strftime(f'%Y-%m-%d-%H%M')
    exp_name = f'{args.experiment}_{args.model_name_or_path}_{day}'
    # WandB init and config (based on argument dictionaries in imports/globals cell)
    wandb.init(
        name=exp_name,
        project='rf-bert',
        entity='jscuds',
        notes=None,
        config=vars(args)
    )

    # Log to a file and stdout
    Path("logs/").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'logs/{exp_name}.log'),
            logging.StreamHandler()
        ])

    # train on gpu if availble, set `device` as global variable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment.model.to(device)
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=args.learning_rate)

    # use wandb.watch() to track gradients
    # watch_log_freq is setup to log every 10 batches: num_examples//batch_size//10
    #    which means it will log gradients every `watch_log_freq` batches

    watch_log_freq = args.num_examples//args.batch_size//10
    wandb.watch(experiment.model, log_freq=watch_log_freq)

    log_interval = max(len(train_dataloader) // args.logs_per_epoch, 1)
    logger.info(f'Logging metrics every {log_interval} training steps')

    epoch_start_time = time.time()
    for epoch in range(args.epochs):
        logger.info(f"Starting training epoch {epoch+1}/{args.epochs}")
        for step, (batch, targets) in tqdm.tqdm(enumerate(train_dataloader), leave=False):
            batch, targets = batch.to(device), targets.to(device) # TODO(js) retrofit_change
            preds = experiment.model(batch) 
            # TODO: cast to float in dataloader and remove this call to float()
            train_loss = experiment.compute_loss_and_update_metrics(preds, targets.float(), 'Train')
            
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_interval == 0:
                logger.info(f"Running evaluation at step {step} in epoch {epoch} (logs_per_epoch = {args.logs_per_epoch})")
                # Compute eval metrics every `log_interval` batches
                experiment.model.eval() # set model in eval mode for evaluation
                with torch.no_grad():
                    for batch, targets in tqdm.tqdm(test_dataloader, leave=False):
                        batch, targets = batch.to(device), targets.to(device)
                        preds = experiment.model(batch)
                        experiment.compute_loss_and_update_metrics(preds, targets.float(), 'Test')
                # Compute metrics, log, and reset
                metrics_dict = experiment.compute_and_reset_metrics()
                wandb.log(metrics_dict)
                # Set model back in train mode to resume training
                experiment.model.train() 
            
        # Log elapsed time at end of each epoch    
        epoch_end_time = time.time()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Total EPOCH Time: {(epoch_end_time-epoch_start_time)/60:>0.2f} min")
        wandb.log({"Epoch Time": (epoch_end_time-epoch_start_time)/60})

        epoch_start_time = time.time()
    logging.info(f'***** Training finished after {args.epochs} epochs *****')


def run_training_loop_retrofit(args: argparse.Namespace):
    # dictionary that matches experiment argument to its respective class
    experiment_cls = {
        'retrofit': RetrofitExperiment,
        'finetune': FinetuneExperiment
    }[args.experiment]
    experiment = experiment_cls(args)

    #########################################################
    ################## DATASET & DATALOADER #################
    #########################################################

    # NOTE(js): HAD TO COPY QUORA BECAUSE CHANGING THE SPLIT CHANGED THE DATALOADERS.

    train_dataset = experiment.dataset
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, pin_memory=True)

    logger.info('\n***CHECK DATASET LENGTHS:***')
    logger.info(f'len(train_dataloader.dataset) = {len(train_dataloader.dataset)}')


    #########################################################
    ###################### TRAINING LOOP ####################
    #########################################################


    day = time.strftime(f'%Y-%m-%d-%H%M')
    exp_name = f'{args.experiment}_{args.model_name_or_path}_{day}'
    # WandB init and config (based on argument dictionaries in imports/globals cell)
    wandb.init(
        name=exp_name,
        project='rf-bert',
        entity='jscuds',
        tags=['cluster',args.model_name_or_path,'rf-loss','sgd'],
        notes="Job ID: <CLUSTER ID HERE> \nAttempt to log 1 time per epoch\nepochs=3\nSGD(batch_size 512);\nLR=1e-5", #'loss.sum()\nlogs_per_epoch=1\nRan with *Adam optimizer* and reported "best" hyperparameters  rf_gamma=3, rf_lambda=1, epochs=10, lr=0.005, BUT batch_size=512'
        config=vars(args)
    )

    # Log to a file and stdout
    Path("logs/").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'logs/{exp_name}.log'),
            logging.StreamHandler()
        ])

    # train on gpu if availble, set `device` as global variable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment.model.to(device)
    optimizer = torch.optim.SGD(experiment.model.parameters(), lr=args.learning_rate) # NOTE(js): manually changed this to Adam/SGD to test retrofit training `optimizer = torch.optim.Adam(experiment.model.parameters(), lr=args.learning_rate)`

    # use wandb.watch() to track gradients
    # watch_log_freq is setup to log every 10 batches: num_examples//batch_size//10
    #    which means it will log gradients every `watch_log_freq` batches

    watch_log_freq = args.num_examples//args.batch_size//10
    wandb.watch(experiment.model, log_freq=watch_log_freq)

    log_interval = max(len(train_dataloader) // args.logs_per_epoch, 1)
    logger.info(f'Logging metrics every {log_interval} training steps')

    epoch_start_time = time.time()
    for epoch in range(args.epochs):
        logger.info(f"Starting training epoch {epoch+1}/{args.epochs}")
        for step, (batch) in tqdm.tqdm(enumerate(train_dataloader), leave=False):
            # TODO(js) retrofit_change
            sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2 = batch
            sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2 = sent1.to(device), sent2.to(device), nsent1.to(device), nsent2.to(device), token1.to(device), token2.to(device), ntoken1.to(device), ntoken2.to(device)
            
            word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2 = experiment.model(sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2) 
            
            train_loss = experiment.compute_loss_and_update_metrics(word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2, 'Train')
            
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % log_interval == 0:
                logger.info(f"Running evaluation at step {step} in epoch {epoch} (logs_per_epoch = {args.logs_per_epoch})")
                # Compute metrics, log, and reset
                metrics_dict = experiment.compute_and_reset_metrics()
                wandb.log(metrics_dict)

            
        # Log elapsed time at end of each epoch    
        epoch_end_time = time.time()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Total EPOCH Time: {(epoch_end_time-epoch_start_time)/60:>0.2f} min")
        wandb.log({"Epoch Time": (epoch_end_time-epoch_start_time)/60})

        epoch_start_time = time.time()
    logging.info(f'***** Training finished after {args.epochs} epochs *****')
    # Save M matrix
    # TODO(js): save whole model?
    torch.save(experiment.model.elmo._elmo_lstm._elmo_lstm.M,f'M_matrix_{device}_{day}.pt')


if __name__ == '__main__':
    args: argparse.Namespace = get_argparser().parse_args()
    set_random_seed(args.random_seed)
    if args.experiment == 'finetune':
        run_training_loop(args)
    if args.experiment == 'retrofit':
        run_training_loop_retrofit(args)

    
# TODO(js): after confirming retrofitting works; fix `run_training_loop()` to work for either case
#   - update Dataset __getitem__ to return dictionary of kwargs
#   - update models to receive kwargs passed from applicable Dataset objects
    