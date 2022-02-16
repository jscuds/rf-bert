import argparse
import json
import logging
import random
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader

from experiment import FinetuneExperiment, RetrofitExperiment

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(r):
    random.seed(r)
    np.random.seed(r)
    torch.manual_seed(r)
    torch.cuda.manual_seed(r)


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('experiment', type=str, choices=('retrofit', 'finetune'))

    parser.add_argument('--optimizer', type=str, default='sgd', choices=('adam', 'sgd'),
        help='Optimizer to use for training')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5,
        help='number of training epochs')
    parser.add_argument('--epochs_per_model_save', default=3, 
        type=int, help='number of epochs between model saves')
    parser.add_argument('--logs_per_epoch', type=int, default=5,
        help='log metrics this number of times per epoch')
    parser.add_argument('--num_examples', type=int, default=None,
        help='number of training examples')
    parser.add_argument('--max_length', type=int, default=40,
        help='max length of each sequence for truncation')
    parser.add_argument('--train_test_split', type=float, default=0.8,
        help='percent of data to use for train, in (0, 1]')

    parser.add_argument('--model_name', type=str, default='elmo_single_sentence', 
        choices=['elmo_single_sentence', 'elmo_sentence_pair'], help='name of model to use')
    parser.add_argument('--dataset_name', type=str, default='qqp', 
        choices=['qqp', 'rotten_tomatoes'], help='name of dataset to use for finetuning')
    parser.add_argument('--model_weights', type=str, default=None,
        help='path to model weights to load, like `models/something.pth`')
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
    parser.add_argument('--finetune_rf', default=False, action='store_true',
        help=('If we are running a fine-tuning experiment, this indicates that we are fine-tuning a model that was '
            'previously retrofitted, so we can load the model with the proper architecture (i.e. including M matrix)'))
    parser.add_argument('--wandb_tags', nargs='+', default=None,
        help='add list of strings to be used as tags for W&B')
    parser.add_argument('--wandb_notes', type=str, default=None,
        help='pass notes for W&B runs.')


    # TODO add dataset so we can switch between 'quora', 'mrpc'...
    # TODO add _task_dataset so we can switch between tasks for evaluation/attack
    # TODO: additional model args? dropout...
    
    return parser

def run_training_loop(args: argparse.Namespace) -> str:
    """Runs model-training experiment defined by `args`.

    Returns:
        model_folder (str): folder with saved final model & checkpoints
    """
    if (args.num_examples is not None) and (args.batch_size > args.num_examples):
        logger.warn("Batch size (%d) cannot be greater than num examples (%d), decreasing batch size", args.batch_size, args.num_examples)
        args.batch_size = args.num_examples
    
    # dictionary that matches experiment argument to its respective class
    experiment_cls = {
        'retrofit': RetrofitExperiment,
        'finetune': FinetuneExperiment
    }[args.experiment]
    experiment = experiment_cls(args)

    # Distribute training across multiple GPUs
    if torch.cuda.device_count() > 1:
        logging.info(f'torch.nn.DataParallel distributing training across {torch.cuda.device_count()} GPUs')
        experiment.model = torch.nn.DataParallel(experiment.model)
    
    print(experiment.model)

    #########################################################
    ################## DATASET & DATALOADER #################
    #########################################################

    train_dataloader, test_dataloader = experiment.get_dataloaders()

    logger.info('\n***CHECK DATALOADER LENGTHS:***')
    logger.info(f'len(train_dataloader) = {len(train_dataloader)}')
    logger.info(f'len(test_dataloader) = {len(test_dataloader)}')


    #########################################################
    ###################### TRAINING LOOP ####################
    #########################################################
    day = time.strftime(f'%Y-%m-%d-%H%M')
    # NOTE(js): `args.model_name[:4]` just grabs "elmo" or "bert"; feel free to change later
    exp_name = f'{args.experiment}_{args.model_name[:4]}_{day}' 
    # WandB init and config (based on argument dictionaries in imports/globals cell)
    config_dict = vars(args) 
    config_dict.update({
        "train_dataloader_len": len(train_dataloader),
        "test_dataloader_len": len(test_dataloader), 
    })
    
    wandb.init(
        name=exp_name,
        project=os.environ.get('WANDB_PROJECT', 'rf-bert'),
        entity=os.environ.get('WANDB_ENTITY', 'jscuds'),
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        config=config_dict,
    )

    # Log to a file and stdout
    Path("logs/").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'logs/{exp_name}.log'),
            logging.StreamHandler()
        ])

    # Create folder for model saves (and parent models/ folder, if needed)
    model_folder = f"models/{exp_name}/"
    Path(model_folder).mkdir(exist_ok=True, parents=True)
    # Log args to folder
    args_file_path = os.path.join(model_folder, 'args.json')
    logging.info('Saving training args to %s', args_file_path)
    with open(args_file_path, 'w') as args_file:
        json.dump(config_dict, args_file)

    # train on gpu if availble, set `device` as global variable
    experiment.model.to(device)
    # load model from disk
    if args.model_weights:
        logging.info('*** loading model %s from path %s', args.model_name, args.model_weights)
        model_weights = torch.load(args.model_weights, map_location=device)
        logging.info('*** loaded model weights from disk, keys = %s', model_weights.keys())
        missing_keys, unexpected_keys = experiment.model.load_state_dict(model_weights['model'], strict=False)
        logging.info('*** loaded model weights into model, missing_keys = %s', missing_keys)
        logging.info('*** loaded model weights into model, unexpected_keys = %s', unexpected_keys)

        # Allow missing keys if it's for the two linear layers from our classifier.
        # This is for fine-tuning models that were pre-trained (retrofitted) without a
        # linear layer.
        # TODO: make this nicer, this is a little hacky.
        if len(missing_keys):
            assert missing_keys == ['linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias'], "unknown missing keys. Maybe you forgot the --finetune_rf argument?"
        # And there should definitely never be any weights we're loading that don't have
        # anywhere to go.
        assert len(unexpected_keys) == 0

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(experiment.model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(experiment.model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f'unsupported optimizer {args.optimizer}')

    # use wandb.watch() to track gradients
    # watch_log_freq is setup to log every 10 batches:
    wandb.watch(experiment.model, log_freq=round(len(train_dataloader) / 10.0))

    log_interval = max(len(train_dataloader) // args.logs_per_epoch, 1)
    logger.info(f'Logging metrics every {log_interval} training steps')

    epoch_start_time = time.time()
    training_step = 0
    for epoch in range(args.epochs):
        logger.info(f"Starting training epoch {epoch+1}/{args.epochs}")
        for _epoch_step, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training', leave=False):
            if args.experiment == 'finetune':
                if len(batch) == 2: # single-sentence classification
                    sentence, targets = batch
                    sentence, targets = sentence.to(device), targets.to(device) # TODO(js) retrofit_change
                    preds = experiment.model(sentence)
                elif len(batch) == 3: # sentence-pair classification
                    sentence1, sentence2, targets = batch
                    # We pass sentence pairs as a tensor of shape (B, 2, ...) instead of a tuple of two tensors.
                    sentence_stacked = torch.stack((sentence1, sentence2), axis=1).to(device)
                    targets = targets.to(device)
                    preds = experiment.model(sentence_stacked)
                else:
                    raise ValueError(f'Expected batch of length 2 or 3, got {len(batch)}')
                train_loss = experiment.compute_loss_and_update_metrics(preds, targets, 'Train')
            else:
                # sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2 = batch
                batch = (t.to(device) for t in batch)
                word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2 = experiment.model(*batch)
                train_loss = experiment.compute_loss_and_update_metrics(word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2, 'Train')
            
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (training_step + 1) % log_interval == 0:
                logger.info(f"Running evaluation at step {training_step} in epoch {epoch+1} (logs_per_epoch = {args.logs_per_epoch})")
                experiment.model.eval() # set model in eval mode for evaluation
                # Compute eval metrics every `log_interval` batches
                with torch.no_grad():
                    for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader), desc='Evaluating', leave=False):
                        if args.experiment == 'finetune':
                            if len(batch) == 2: # single-sentence classification
                                sentence, targets = batch
                                sentence, targets = sentence.to(device), targets.to(device) # TODO(js) retrofit_change
                                preds = experiment.model(sentence)
                            elif len(batch) == 3: # sentence-pair classification
                                sentence1, sentence2, targets = batch
                                # We pass sentence pairs as a tensor of shape (B, 2, ...) instead of a tuple of two tensors.
                                sentence_stacked = torch.stack((sentence1, sentence2), axis=1).to(device)
                                targets = targets.to(device)
                                preds = experiment.model(sentence_stacked)
                            else:
                                raise ValueError(f'Expected batch of length 2 or 3, got {len(batch)}')
                            experiment.compute_loss_and_update_metrics(preds, targets, 'Test')
                        else:
                            # sent1, sent2, nsent1, nsent2, token1, token2, ntoken1, ntoken2 = batch
                            batch = (t.to(device) for t in batch)
                            word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2 = experiment.model(*batch) 
                            experiment.compute_loss_and_update_metrics(word_rep_pos_1, word_rep_pos_2, word_rep_neg_1, word_rep_neg_2, 'Test')
                # Compute metrics, log, and reset
                metrics_dict = experiment.compute_and_reset_metrics(training_step, epoch)
                wandb.log(metrics_dict, step=training_step)
                # Set model back in train mode to resume training
                experiment.model.train() 
            
            # End of epoch step, increment counter
            training_step += 1
            
        if (epoch+1) % args.epochs_per_model_save == 0:
            checkpoint = {
                'step': training_step, 
                'model': experiment.model.state_dict()
            }
            checkpoint_path = os.path.join(
                model_folder, f'{epoch}_epochs.pth')   
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model checkpoint saved to %s after epoch %d', checkpoint_path, epoch)
            
        # Log elapsed time at end of each epoch    
        epoch_end_time = time.time()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Total EPOCH Time: {(epoch_end_time-epoch_start_time)/60:>0.2f} min")
        wandb.log({"Epoch Time": (epoch_end_time-epoch_start_time)/60})

        epoch_start_time = time.time()
    logging.info(f'***** Training finished after {args.epochs} epochs *****')

    # Save final model
    final_save_path = os.path.join(
        model_folder, f'final.pth')   
    torch.save({ 'model': experiment.model.state_dict() }, final_save_path)
    logging.info('Final model saved to %s', final_save_path)

    return model_folder


if __name__ == '__main__':
    args: argparse.Namespace = get_argparser().parse_args()
    set_random_seed(args.random_seed)
    run_training_loop(args)

    
# TODO(js): after confirming retrofitting works; fix `run_training_loop()` to work for either case
#   - update Dataset __getitem__ to return dictionary of kwargs
#   - update models to receive kwargs passed from applicable Dataset objects
    