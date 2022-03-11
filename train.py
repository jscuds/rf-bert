import argparse
import copy
import json
import logging
import os
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
    parser.add_argument('--save_model_on_eval', default=False, 
        action='store_true', help='save model on eval step')
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
    parser.add_argument('--rf_dataset_name', type=str, default='quora', 
        choices=['quora', 'mrpc'], help='name of dataset to use for retrofitting')
    parser.add_argument('--ft_dataset_name', type=str, default='qqp', 
        choices=['qqp', 'rotten_tomatoes', 'sst2'], help='name of dataset to use for finetuning')
    parser.add_argument('--model_weights', type=str, default=None,
        help='path to model weights to load, like `models/something.pth`')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--rf_lambda', type=float, default=1,
        help='lambda - regularization constant for retrofitting loss')
    parser.add_argument('--rf_gamma', type=float, default=2,
        help='gamma - margin constant for retrofitting loss')
    parser.add_argument('--num_table_examples', type=int, default=None,
        help='examples to watch in both train/test sets - will log to W&B Table')
    parser.add_argument('--elmo_dropout', type=float, default=0.0,
        help='dropout probability (0,1] for ELMO embeddings model')
    parser.add_argument('--ft_dropout', type=float, default=0.2,
        help='dropout probability (0,1] for classifier model')

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
    parser.add_argument('--model_weights_drop_linear', default=False, action='store_true',
        help='If we are fine-tuning a model that previously had a linear layer, don\'t load the weights for the linear layer')
    parser.add_argument('--wandb_tags', nargs='+', default=None,
        help='add list of strings to be used as tags for W&B')
    parser.add_argument('--wandb_notes', type=str, default=None,
        help='pass notes for W&B runs.')


    # TODO add dataset so we can switch between 'quora', 'mrpc'...
    # TODO add _task_dataset so we can switch between tasks for evaluation/attack
    
    return parser

def run_training_loop(args: argparse.Namespace) -> str:
    """Runs model-training experiment defined by `args`.

    Returns:
        model_folder (str): folder with saved final model & checkpoints
    """
    if (args.num_examples is not None) and (args.batch_size > args.num_examples):
        logger.warn("Batch size (%d) cannot be greater than num examples (%d), decreasing batch size", args.batch_size, args.num_examples)
        args.batch_size = args.num_examples
    
    if (args.num_table_examples is not None) and (args.num_table_examples > args.batch_size):
        logger.warn("Batch size (%d) cannot be greater than num table examples (%d), decreasing num table examples", args.batch_size, args.num_table_examples)
        args.num_table_examples = args.batch_size

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
    config_dict = copy.copy(vars(args))
    config_dict.update({
        "train_dataloader_len": len(train_dataloader),
        "test_dataloader_len": len(test_dataloader), 
    })
    # Remove these args from the W&B config listing to prevent clutter
    if args.wandb_tags:
        del config_dict['wandb_tags']
    if args.wandb_notes: 
        del config_dict['wandb_notes']
    
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
        force=True,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'logs/{exp_name}.log'),
            logging.StreamHandler()
        ]
    )

    # Create folder for model saves (and parent models/ folder, if needed)
    model_folder = f"models/{exp_name}/"
    Path(model_folder).mkdir(exist_ok=True, parents=True)
    # Log args to folder
    args_file_path = os.path.join(model_folder, 'args.json')
    logging.info('Saving training args to %s', args_file_path)
    with open(args_file_path, 'w') as args_file:
        json.dump(config_dict, args_file)

    # load model from disk
    if args.model_weights:
        logging.info('*** loading model %s from path %s', args.model_name, args.model_weights)
        model_weights = torch.load(args.model_weights, map_location=device)['model']
        logging.info('*** loaded model weights from disk, keys = %s', model_weights.keys())

        if args.model_weights_drop_linear:
            # Say we want to fine-tune a previously fine-tuned model and drop its linear layers.
            # We can do so 
            logging.info(f'--model_weights_drop_linear set, not loading weights for linear layers')
            linear_keys = [k for k in model_weights.keys() if k.startswith('linear')]

            if not len(linear_keys):
                raise ValueError('--model_weights_drop_linear set but no weights for linear layers found')

            for key in linear_keys:
                logging.info(f'\t- dropping weight with key `%s`', key)
                del model_weights[key]

        missing_keys, unexpected_keys = experiment.model.load_state_dict(model_weights, strict=False)
        logging.info('*** loaded model weights into model, missing_keys = %s', missing_keys)
        logging.info('*** loaded model weights into model, unexpected_keys = %s', unexpected_keys)

        # Allow missing keys if it's for the two linear layers from our classifier.
        # This is for fine-tuning models that were pre-trained (retrofitted) without a
        # linear layer.
        # TODO: make this nicer, this is a little hacky.
        if len(missing_keys):
            assert missing_keys == ['classifier.1.weight', 'classifier.1.bias', 'classifier.4.weight', 'classifier.4.bias'], "unknown missing keys. Maybe you forgot the --finetune_rf argument?"
        # And there should definitely never be any weights we're loading that don't have
        # anywhere to go.
        if len(unexpected_keys):
            assert unexpected_keys == ['elmo.scalar_mix_1.gamma', 'elmo.scalar_mix_1.scalar_parameters.0', 'elmo.scalar_mix_1.scalar_parameters.1', 'elmo.scalar_mix_1.scalar_parameters.2', 'elmo.scalar_mix_2.gamma', 'elmo.scalar_mix_2.scalar_parameters.0', 'elmo.scalar_mix_2.scalar_parameters.1', 'elmo.scalar_mix_2.scalar_parameters.2', 'elmo.scalar_mix_3.gamma', 'elmo.scalar_mix_3.scalar_parameters.0', 'elmo.scalar_mix_3.scalar_parameters.1', 'elmo.scalar_mix_3.scalar_parameters.2']

    # use wandb.watch() to track gradients
    # watch_log_freq is setup to log every 10 batches:
    wandb.watch(experiment.model, log_freq=round(len(train_dataloader) / 10.0))

    log_interval = max(len(train_dataloader) // args.logs_per_epoch, 1)
    logger.info(f'Logging metrics every {log_interval} training steps')

    epoch_start_time = time.time()
    training_step = 0
    # TODO: Optionally run an eval step before training starts (to get the model
    # stats before any weights are changed)
    for epoch in range(args.epochs):
        logger.info(f"Starting training epoch {epoch+1}/{args.epochs}")
        for epoch_step, train_batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training', leave=False):
            is_first_batch = (epoch_step == 0)
            train_loss = experiment.compute_loss_and_update_metrics(
                train_batch, metrics_key='Train', epoch=epoch, is_first_batch=is_first_batch)
            experiment.backward(train_loss, epoch=epoch)
            
            if (training_step + 1) % log_interval == 0:
                logger.info(f"Running evaluation at step {training_step} in epoch {epoch+1} (logs_per_epoch = {args.logs_per_epoch})")
                experiment.model.eval() # set model in eval mode for evaluation
                # Compute eval metrics every `log_interval` batches
                with torch.no_grad():
                    for test_batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader), desc='Evaluating', leave=False):
                        experiment.compute_loss_and_update_metrics(
                            test_batch, metrics_key='Test', epoch=epoch, is_first_batch=is_first_batch)
                # Advance learning rate scheduler (if there is one) after validation.
                # (Note: we have to do this *before* logging metrics so test loss is not zero.)
                experiment.step_lr_scheduler()
                # Compute metrics, log, and reset
                metrics_dict = experiment.compute_and_reset_metrics(step=training_step, epoch=epoch)
                wandb.log(metrics_dict, step=training_step)
                # Compute min and max summary for each metric.
                for metric, val in metrics_dict.items():
                    min_key = f"{metric}_min"
                    wandb.run.summary[min_key] = min(
                        wandb.run.summary.get(min_key, val), val)
                    max_key = f"{metric}_max"
                    wandb.run.summary[max_key] = max(
                        wandb.run.summary.get(max_key, val), val)
                # Set model back in train mode to resume training
                experiment.model.train() 
                # Save model, if specified
                if args.save_model_on_eval:
                    checkpoint = {
                        'step': training_step, 
                        'model': experiment.model.state_dict()
                    }
                    checkpoint_path = os.path.join(
                        model_folder, f'save_step_{training_step}.pth')   
                    torch.save(checkpoint, checkpoint_path)
                    logging.info('Model checkpoint saved to %s after eval at step %d', checkpoint_path, training_step)
            # End of step, increment counter
            training_step += 1
        # End of epoch
        if (epoch+1) % args.epochs_per_model_save == 0:
            checkpoint = {
                'step': training_step, 
                'model': experiment.model.state_dict()
            }
            checkpoint_path = os.path.join(
                model_folder, f'save_epoch_{epoch}.pth')   
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model checkpoint saved to %s after epoch %d', checkpoint_path, epoch)
            
        # Log elapsed time at end of each epoch    
        epoch_end_time = time.time()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Total EPOCH Time: {(epoch_end_time-epoch_start_time)/60:>0.2f} min")
        wandb.log({"Epoch Time": (epoch_end_time-epoch_start_time)/60})

        epoch_start_time = time.time()

    logging.info(f'***** Training finished after {args.epochs} epochs *****')

    # After training, log example table
    if args.experiment == 'retrofit' and args.num_table_examples is not None:
        experiment.wb_table.update_wandb_table()
        wandb.log({'sampled_examples_table': experiment.wb_table.final_table}) 

    # Save final model
    final_save_path = os.path.join(model_folder, 'final.pth')   
    torch.save({ 'model': experiment.model.state_dict() }, final_save_path)
    logging.info('Final model saved to %s', final_save_path)

    return model_folder


if __name__ == '__main__':
    args: argparse.Namespace = get_argparser().parse_args()
    set_random_seed(args.random_seed)
    run_training_loop(args)

    
# TODO(js):
#   - update Dataset __getitem__ to return dictionary of kwargs
#   - update models to receive kwargs passed from applicable Dataset objects
    
