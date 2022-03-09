import glob
import os
import shutil

import pytest

from train import get_argparser, run_training_loop


class TestTrainEnd2End:
    """End-to-end testing for training loop in train.py."""
    def test_finetune_qqp_tiny(self):
        os.environ['WANDB_MODE'] = 'disabled' # disable W&B for testing
        with pytest.raises(AssertionError):
            # This will fail because we specify the `elmo_single_sentence` model but
            # the QQP data is pairs of sentences.
            args = get_argparser().parse_args(
                ['finetune', '--model_name', 'elmo_single_sentence', '--epochs', '2', '--num_examples', '8', '--batch_size', '4', '--ft_dataset_name', 'qqp']
            )
            model_folder = run_training_loop(args)
            shutil.rmtree(model_folder) # delete model saves after test
        # Now try again, but use the sentence-pair model.
        args = get_argparser().parse_args(
            ['finetune', '--model_name', 'elmo_sentence_pair', '--epochs', '2', '--num_examples', '8', '--batch_size', '4', '--ft_dataset_name', 'qqp']
        )
        model_folder = run_training_loop(args)
        shutil.rmtree(model_folder) # delete model saves after test
        

    def test_finetune_rottentomatoes_tiny(self):
        os.environ['WANDB_MODE'] = 'disabled' # disable W&B for testing

        args = get_argparser().parse_args(
            ['finetune', '--model_name', 'elmo_single_sentence', '--epochs', '2', '--num_examples', '8', '--batch_size', '4', '--ft_dataset_name', 'rotten_tomatoes']
        )
        model_folder = run_training_loop(args)
        shutil.rmtree(model_folder) # delete model saves after test


    def test_retrofit_quora_tiny(self):
        os.environ['WANDB_MODE'] = 'disabled' # disable W&B for testing

        args = get_argparser().parse_args(
            ['retrofit', '--epochs', '2', '--num_examples', '8', '--batch_size', '4']
        )
        model_folder = run_training_loop(args)
        shutil.rmtree(model_folder) # delete model saves after test

    
    def test_retrofit_and_then_finetune(self):
        os.environ['WANDB_MODE'] = 'disabled' # disable W&B for testing

        # Run a little bit of retrofit training 
        args1 = get_argparser().parse_args(
            ['retrofit', '--epochs', '2', '--num_examples', '8', '--batch_size', '4']
        )
        rf_model_folder = run_training_loop(args1)
        rf_model_path = glob.glob(rf_model_folder + '/*.pth')[0]
        # Get path to an actual model

        # Try fine-tuning this model, should fail without the
        # --finetune_rf flag, when we try to load weights of a model
        # with an M matrix into a model that doesn't have an M matrix
        args2 = get_argparser().parse_args(
            ['finetune', '--model_name', 'elmo_single_sentence', '--ft_dataset_name', 'rotten_tomatoes', '--epochs', '2', '--num_examples', '8', '--batch_size', '4', '--model_weights', rf_model_path]
        )
        with pytest.raises(AssertionError):
            run_training_loop(args2)

        # Now actually fine-tune that model for a tiny bit
        args3 = get_argparser().parse_args(
            ['finetune', '--model_name', 'elmo_single_sentence', '--ft_dataset_name', 'rotten_tomatoes', '--epochs', '2', '--num_examples', '8', '--batch_size', '4', '--model_weights', rf_model_path, '--finetune_rf']
        )
        ft_model_folder = run_training_loop(args3)
        # Delete new and old models
        shutil.rmtree(rf_model_folder)
        shutil.rmtree(ft_model_folder)

