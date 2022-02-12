import os
import shutil

import pytest

from train import get_argparser, run_training_loop


class TestTrainEnd2End:
    """End-to-end testing for training loop in train.py."""
    def test_finetune_quora_tiny(self):
        os.environ['WANDB_MODE'] = 'disabled' # disable W&B for testing

        args = get_argparser().parse_args(
            ['finetune', '--epochs', '2', '--num_examples', '8']
        )
        model_folder = run_training_loop(args)
        shutil.rmtree(model_folder) # delete model saves after test


    def test_retrofit_quora_tiny(self):
        os.environ['WANDB_MODE'] = 'disabled' # disable W&B for testing

        args = get_argparser().parse_args(
            ['retrofit', '--epochs', '2', '--num_examples', '8']
        )
        model_folder = run_training_loop(args)
        shutil.rmtree(model_folder) # delete model saves after test

