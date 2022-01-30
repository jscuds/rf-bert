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

from .experiments import FinetuneExperiment, RetrofitExperiment

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
    parser.add_argument('--num_examples', type=int, default=20_000,
        description='number of training examples')
    parser.add_argument('--train_test_split', type=float, default=0.8,
        description='percent of data to use for train, in (0, 1]')

    parser.add_argument('--model_name_or_path', default='elmo', 
        choices=['elmo'], description='name of model to use')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--drop_last', type=bool, default=False,
        description='whether to drop remainder of last batch')

    args = parser.parse_args()
    set_random_seed(args.random_seed)

    day = time.strftime(f'%Y-%m-%d')

    # NUM_EXAMPLES = 20000 #20000 #100000  404290   #EXP CHANGE 1
    # EVAL_METHOD = 'GLUE' #base #GLUE
    # METHOD = f'SIZE_{NUM_EXAMPLES//1000}k_{EVAL_METHOD}' #100k #20k  #full  #50-50  #no-repeat  #filter
    # TITLE = f'quora-{METHOD}_{day}'

    # MODEL_NAME = 
    # MAX_LENGTH = 40 #elmo_change # *2 FOR BERT multiply by 2 because now passing two paraphrases together

    # TRAIN_SIZE = 0.8
    # TEST_SIZE = 0.2
    # RNDM_SEED = 42
    # STRAT = False #EXP CHANGE 2
    # CHECK_DUP = False #EXP CHANGE 3

    # LR = 1e-4
    # NUM_EPOCHS = 6 # elmo_change
    # BATCH_SIZE = 512
    # DROP_LAST = False
    
    return args

def main():

    args = parse_args()

    experiment_cls = {
        'retrofit': RetrofitExperiment,
        'finetune': FinetuneExperiment
    }[args.experiment]
    experiment = experiment_cls(args)

    #########################################################
    ################## DATASET & DATALOADER #################
    #########################################################

    # js NOTE: HAD TO COPY QUORA BECAUSE CHANGING THE SPLIT CHANGED THE DATALOADERS.
    quora.split('train')
    train_quora = copy.copy(quora)
    train_dataloader = DataLoader(train_quora, batch_size=BATCH_SIZE, shuffle=True, drop_last=DROP_LAST, pin_memory=True)
    quora.split('test')
    test_quora = copy.copy(quora)
    test_dataloader = DataLoader(test_quora, batch_size=BATCH_SIZE, shuffle=True, drop_last=DROP_LAST, pin_memory=True)

    print('\n***CHECK DATASET LENGTHS:***')
    print(f'len(train_dataloader.dataset) = {len(train_dataloader.dataset)}')
    print(f'len(test_dataloader.dataset) = {len(test_dataloader.dataset)}')

    print('\n***CHECK self.tokenized_sentences.shape: ')
    print(f'quora.tokenized_sentences.shape: {quora.tokenized_sentences.shape}\nintended shape: {(quora.num_examples,2,40,50)}')


    #########################################################
    ###################### TRAINING LOOP ####################
    #########################################################


    # WandB init and config (based on argument dictionaries in imports/globals cell)
    wandb.init(
        name=TITLE,
        project='rf-bert',
        entity='jscuds',
        notes=None,
        config=dict(args)
    )

    # def compute_n_correct(preds, targets):
    #     return torch.sum((torch.sigmoid(preds.squeeze()).round() == targets)).cpu().item()

    # train on gpu if availble, set `device` as global variable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = BCEWithLogitsLoss()

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
    for epoch in range(NUM_EPOCHS):
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
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Loss: {train_losses[-1]:>0.5f}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Accuracy: {train_accs[-1]*100:>0.2f}%" )
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train F1: {train_f1[-1]*100:>0.2f}" )
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Prec: {train_precision[-1]:>0.2f}" )
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Recall: {train_recall[-1]:>0.2f}" )


        running_loss = 0
        n_correct = 0
        preds_accum = []
        targets_accum = []

        with torch.no_grad():
            for batch, targets in tqdm(test_dataloader, leave=False):
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
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Loss: {test_losses[-1]:>0.5f}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Accuracy: {test_accs[-1]*100:>0.2f}%" )
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test F1: {test_f1[-1]*100:>0.2f}" )
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Prec: {test_precision[-1]:>0.2f}" )
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Test Recall: {test_recall[-1]:>0.2f}" )
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Total EPOCH Time: {(t1-t0)/60:>0.2f} min")
        

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