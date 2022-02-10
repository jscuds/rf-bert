import argparse
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
from utils import train_test_split

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
    parser.add_argument('--num_examples', type=int, default=25_000,
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

    # for these boolean arguments, append the flag if you want it to be `True`
    #     otherwise, omit the flag if you want it to be False
    #     ex1.  `python train.py --drop_last` will drop the remainder of the last batch in the dataloaders.
    #     ex2.  `python train.py --req_grad_elmo`` will make ELMo weights update
    #     Refs: https://stackoverflow.com/questions/52403065/argparse-optional-boolean
    #           https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument('--drop_last', default=False, action='store_true',
        help='whether to drop remainder of last batch')
    parser.add_argument('--req_grad_elmo', default=False, action='store_true',
        help='ELMo requires_grad means don\'t freeze weights during retrofit training')


    # TODO add dataset so we can switch between 'quora', 'mrpc'...
    # TODO add _task_dataset so we can switch between tasks for evaluation/attack
    # TODO: additional args? dropout...
    
    return parser


def create_wandb_histogram(list_of_values: list, description: str, epoch: int):
    lower_description = '_'.join(description.lower().split())
    data = [[val] for val in list_of_values]
    table = wandb.Table(data=data, columns=[lower_description])
    wandb.log({f'{lower_description}_epoch_{epoch+1}': wandb.plot.histogram(table,lower_description,
                title=f"{description} Histogram, Epoch {epoch+1}")})


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

    train_dataloader, test_dataloader =  train_test_split(experiment.dataset, batch_size=args.batch_size, 
                                                          shuffle=True, drop_last=args.drop_last, 
                                                          train_split=args.train_test_split, seed=args.random_seed)


    logger.info('\n***CHECK DATALOADER LENGTHS:***')
    logger.info(f'len(train_dataloader) = {len(train_dataloader)}')
    logger.info(f'len(test_dataloader) = {len(test_dataloader)}')


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

    train_dataloader, test_dataloader =  train_test_split(experiment.dataset, batch_size=args.batch_size, 
                                                          shuffle=True, drop_last=args.drop_last, 
                                                          train_split=args.train_test_split, seed=args.random_seed)


    logger.info('\n***CHECK DATALOADER LENGTHS:***')
    logger.info(f'len(train_dataloader) = {len(train_dataloader)}')
    logger.info(f'len(test_dataloader) = {len(test_dataloader)}')


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
        tags=['cluster',args.model_name_or_path,'rf-loss','sgd'], #UNFROZEN_ELMO if necessary
        notes="Job ID: <CLUSTER ID HERE> \nQUICK RUN TO VALIDATE TRAIN/TEST SPLIT WORKING.\nelmo frozen\nepochs=3\ngamma=3\nSGD(batch_size 128);\nLR=5e-3", #'loss.sum()\nlogs_per_epoch=1\nRan with *Adam optimizer* and reported "best" hyperparameters  rf_gamma=3, rf_lambda=1, epochs=10, lr=0.005, BUT batch_size=512'
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
                experiment.model.eval()
                with torch.no_grad():
                    for batch in tqdm.tqdm(test_dataloader, leave=False):
                        sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2 = batch
                        sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2 = sent1.to(device), sent2.to(device), nsent1.to(device), nsent2.to(device), token1.to(device), token2.to(device), ntoken1.to(device), ntoken2.to(device)
                        
                        word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2 = experiment.model(sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2) 
                        
                        experiment.compute_loss_and_update_metrics(word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2, 'Test')
                # Compute metrics, log, and reset
                metrics_dict = experiment.compute_and_reset_metrics()
                wandb.log(metrics_dict)
                # Set model back in train mode to resume training
                experiment.model.train() 

        #### BEGIN WANDB HISTOGRAM CODE ####
        # only plot for training data (see experiment.py --> lists only populate if model is training)
        create_wandb_histogram(experiment.pos_dist_list, "Positive Pair Distance", epoch)
        create_wandb_histogram(experiment.neg_dist_list, "Negative Pair Distance", epoch)
        create_wandb_histogram(experiment.diff_dist_list, "Positive minus Negative Pair Distance", epoch)
        create_wandb_histogram(experiment.diff_dist_plus_margin_list, "Positive minus Negative Pair Dist plus Margin", epoch)

        # DON'T FORGET TO RESET LISTS FOR STORING DISTANCES!!
        logger.info(f'\nResetting experiment.pos_dist_list, .neg_dist_list, .diff_dist_list, .diff_dist_plus_margin_list for new histograms')
        experiment.pos_dist_list = []
        experiment.neg_dist_list = []
        experiment.diff_dist_list = [] #pos_dist - neg_dist
        experiment.diff_dist_plus_margin_list = [] #pos_dist + gamma - neg_dist
        logger.info(f'''\nexperiment.pos_dist_list  = {experiment.pos_dist_list}\n experiment.neg_dist_list = {experiment.neg_dist_list}
                        \nexperiment.diff_dist_list = {experiment.diff_dist_list}\nexperiment.diff_dist_plus_margin_list = {experiment.diff_dist_plus_margin_list}''')
        #### END WANDB HISTOGRAM CODE ####


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
    