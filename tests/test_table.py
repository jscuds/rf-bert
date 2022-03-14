import torch
import tqdm
from experiment import RetrofitExperiment
from train import get_argparser, set_random_seed


class TestTableLog:

    def test_wb_table_final_data(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = get_argparser().parse_args(
                    ['retrofit', '--epochs', '1', '--num_examples', '6', '--batch_size', '2', '--random_seed','42', '--num_table_examples','1']
                )
        set_random_seed(args.random_seed)
        experiment = RetrofitExperiment(args)

        train_dataloader, test_dataloader= experiment.get_dataloaders()

        ######### MIMICS RETROFIT TRAINING LOOP #########
        # Given the argparse settings, there should be 4 examples for training, and 2 for testing.
        # With `args.num_table_examples == 1`, there will be 4 rows in experiment.wb_table:
        #     2 rows will correspond to the pos/neg case for the _training_ set
        #     2 rows will correspond to the pos/neg case for the _test_ set
        
        for epoch in range(args.epochs):
            for train_step, train_batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training'):
                # Only save num_table_examples from first batch
                experiment.compute_loss_and_update_metrics(
                    train_batch, metrics_key='Train', epoch=epoch, is_first_batch=(train_step == 0)
                )
            
            experiment.model.eval()
            with torch.no_grad():
                for test_step, test_batch in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Evaluating'):
                    experiment.compute_loss_and_update_metrics(
                        test_batch, metrics_key='Test', epoch=epoch, is_first_batch=(test_step == 0)
                    )
            experiment.model.train()
        
        # Update table at the end of training.
        experiment.wb_table.update_wandb_table()

        # Given a tiny number of total examples (6), and random seed of 42, these are the determined values in the table
        #    I don't check the word distance or individual loss due to floating point error in checking tensors.
        if experiment.wb_table is not None:
            test_column_list = ['epoch', 'index', 'positive/negative', 'sent1', 'sent2', 'shared_word', 'word_distance', 'hinge_loss', 'split']
            test_row_list = list([
                [0, 0, 'positive', 'Where can I get india vs spain davis cup 2016 tickets ?', 'How can I get tickets for the India Spain Davis Cup fixtures ?', 'tickets', 'train'],
                [0, 0, 'negative', 'What are the pros and cons of being a driver for Uber or Lyft in Philadelphia ?', 'What are the pros and cons of being a driver for Uber or Lyft in Seattle ?', 'Uber', 'train'],
                [0, 0, 'positive', 'How can I begin a successful project ?', 'How do I start a successful project ?', 'project', 'validation'],
                [0, 0, 'negative', 'What is the minimum wage in singapore ?', 'What is the minimum wage for IT sales in Singapore ?', 'minimum', 'validation']
            ])

            # Check columns of table
            assert experiment.wb_table.final_table.columns == test_column_list

            # Check all columns that aren't tensors:
            # ['epoch', 'index', 'positive/negative', 'sent1', 'sent2', 'shared_word', 'word_distance', 'hinge_loss', 'split']
            #    not 'word_distance' or 'hinge_loss'
            for i,data_row in enumerate(experiment.wb_table.final_table.data):
                assert data_row[:6] == test_row_list[i][:6]
                assert data_row[-1] == test_row_list[i][-1]