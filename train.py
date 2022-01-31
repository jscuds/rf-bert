import argparse
import copy
import random
import time

import numpy as np
import torch
import tqdm
import wandb

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from experiment import FinetuneExperiment, RetrofitExperiment

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
    parser.add_argument('--num_examples', type=int, default=20_000,
        help='number of training examples')
    parser.add_argument('--train_test_split', type=float, default=0.8,
        help='percent of data to use for train, in (0, 1]')

    parser.add_argument('--model_name_or_path', default='elmo', 
        choices=['elmo'], help='name of model to use')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--drop_last', type=bool, default=False,
        help='whether to drop remainder of last batch')

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

    model = experiment.model

    #########################################################
    ################## DATASET & DATALOADER #################
    #########################################################

    # js NOTE: HAD TO COPY QUORA BECAUSE CHANGING THE SPLIT CHANGED THE DATALOADERS.

    # TODO: refactor this .split() thing, it's not ideal
    train_dataset = copy.copy(experiment.dataset)
    test_dataset = copy.copy(experiment.dataset)

    train_dataset.split('train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, pin_memory=True)
    test_dataset.split('test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, pin_memory=True)

    print('\n***CHECK DATASET LENGTHS:***')
    print(f'len(train_dataloader.dataset) = {len(train_dataloader.dataset)}')
    print(f'len(test_dataloader.dataset) = {len(test_dataloader.dataset)}')

    print('\n***CHECK self.tokenized_sentences.shape: ')
    # print(f'quora.tokenized_sentences.shape: {quora.tokenized_sentences.shape}\nintended shape: {(quora.num_examples,2,40,50)}')


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


    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = experiment.compute_loss

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    train_f1 = []
    test_f1 = []
    train_precision = []
    test_precision = []
    train_recall = []
    test_recall = []
    # TODO: cleanup metric computation

    t0 = time.time()
    for epoch in range(args.epochs):
        running_loss = 0
        n_correct = 0
        preds_accum = []
        targets_accum = []

        for batch, targets in tqdm.tqdm(train_dataloader, leave=False):
            targets_accum.append(targets)
            batch, targets = batch.to(device), targets.to(device)
            preds = model(batch)

            loss = loss_fn(preds.squeeze(), targets.float())

            preds_sigmoid = torch.sigmoid(preds.squeeze())
            preds_accum.append(preds_sigmoid.cpu().detach().numpy().round()) 
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.cpu().item()


        # https://discuss.pytorch.org/t/calculating-f1-score-over-batched-data/83348/2    
        preds_accum = np.concatenate(preds_accum)
        targets_accum = np.concatenate(targets_accum)
        epoch_f1 = f1_score(targets_accum, preds_accum, average='binary')
        epoch_precision = precision_score(targets_accum, preds_accum, average='binary')
        epoch_recall = recall_score(targets_accum, preds_accum, average='binary')
        epoch_accuracy = accuracy_score(targets_accum, preds_accum)
        

        train_losses.append(running_loss / len(train_dataloader.dataset))
        train_accs.append(epoch_accuracy)
        train_f1.append(epoch_f1)
        train_precision.append(epoch_precision)
        train_recall.append(epoch_recall)

        t1 = time.time()
        print("="*20)
        print(f"Epoch {epoch+1}/{args.epochs} Train Loss: {train_losses[-1]:>0.5f}")
        print(f"Epoch {epoch+1}/{args.epochs} Train Accuracy: {train_accs[-1]*100:>0.2f}%" )
        print(f"Epoch {epoch+1}/{args.epochs} Train F1: {train_f1[-1]*100:>0.2f}" )
        print(f"Epoch {epoch+1}/{args.epochs} Train Prec: {train_precision[-1]:>0.2f}" )
        print(f"Epoch {epoch+1}/{args.epochs} Train Recall: {train_recall[-1]:>0.2f}" )


        running_loss = 0
        n_correct = 0
        preds_accum = []
        targets_accum = []

        with torch.no_grad():
            for batch, targets in tqdm.tqdm(test_dataloader, leave=False):
                # for batch, targets in test_dl:
                targets_accum.append(targets)
                batch, targets = batch.to(device), targets.to(device)
                preds = model(batch)
                loss = loss_fn(preds.squeeze(), targets.float())
                preds_sigmoid = torch.sigmoid(preds.squeeze())
                preds_accum.append(preds_sigmoid.cpu().round())

                running_loss += loss.cpu().item()


        preds_accum = np.concatenate(preds_accum)
        targets_accum = np.concatenate(targets_accum)
        epoch_f1 = f1_score(targets_accum, preds_accum, average='binary')
        epoch_precision = precision_score(targets_accum, preds_accum, average='binary')
        epoch_recall = recall_score(targets_accum, preds_accum, average='binary')
        epoch_accuracy = accuracy_score(targets_accum, preds_accum)
        
        test_losses.append(running_loss / len(test_dataloader.dataset))
        test_accs.append(epoch_accuracy)
        test_f1.append(epoch_f1)
        test_precision.append(epoch_precision)
        test_recall.append(epoch_recall)

        t1 = time.time()
        print(f"Epoch {epoch+1}/{args.epochs} Test Loss: {test_losses[-1]:>0.5f}")
        print(f"Epoch {epoch+1}/{args.epochs} Test Accuracy: {test_accs[-1]*100:>0.2f}%" )
        print(f"Epoch {epoch+1}/{args.epochs} Test F1: {test_f1[-1]*100:>0.2f}" )
        print(f"Epoch {epoch+1}/{args.epochs} Test Prec: {test_precision[-1]:>0.2f}" )
        print(f"Epoch {epoch+1}/{args.epochs} Test Recall: {test_recall[-1]:>0.2f}" )
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Total EPOCH Time: {(t1-t0)/60:>0.2f} min")
        

        # WandB log loss, accuracy, and F1 at end of each epoch
        wandb.log({"Train/Loss": train_losses[-1],
                "Train/Acc": train_accs[-1],
                "Train/F1": train_f1[-1],
                "Train/Recall": train_recall[-1],
                "Train/Precision": train_precision[-1],
                "Test/Loss": test_losses[-1],
                "Test/Acc": test_accs[-1],
                "Test/F1": test_f1[-1],
                "Test/Recall": test_recall[-1],
                "Test/Precision": test_precision[-1],
                "Epoch Time": (t1-t0)/60})

        t0 = time.time()

if __name__ == '__main__':
    main()