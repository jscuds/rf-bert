import os

import torch

from experiment import RetrofitExperiment
from train import get_argparser


class TestRetrofitLoss:

    os.environ['WANDB_MODE'] = 'disabled' # disable W&B for testing

    args = get_argparser().parse_args(
        ['retrofit', '--epochs', '2', '--num_examples', '1'] #['finetune', '--epochs', '2', '--num_examples', '8']
    )
    experiment = RetrofitExperiment(args)

    def test_orthogonality_small(self):
        """orthogonality loss should be zero for small orthogonal matrix"""
        M = torch.eye(3, dtype=float)
        assert self.experiment.orthogonalization_loss(M) == 0

    
    def test_orthogonality_big(self):
        """orthogonality loss should be zero for big orthogonal matrix"""
        M = torch.eye(512, dtype=float)
        assert self.experiment.orthogonalization_loss(M) == 0
    

    def test_orthogonality_negate(self):
        """orthogonality loss should be nonzero for a matrix of ones
        with zeros along the diagonal
        """
        M = (
            torch.ones(256, 256, dtype=float) - 
            torch.eye(256, dtype=float)
        )
        assert self.experiment.orthogonalization_loss(M) > 0
    

    def test_orthogonality_rand(self):
        """orthogonality loss should be nonzero for a random matrix
        (unless we randomly get all zeros, which has such a tiny chance
        of happening that let's not worry about that)
        """
        M = torch.rand((128, 128), dtype=float)
        assert self.experiment.orthogonalization_loss(M) > 0
    
    def test_retrofit_hinge_equal(self):
        """when the positive pair are equal, and the negative pair
        are equal, and gamma is zero, loss should be zero
        """
        a = torch.tensor([[1, 0, 1, 0]], dtype=float)
        b = torch.tensor([[0, 1, 0, 1]], dtype=float)
        assert self.experiment.retrofit_hinge_loss(
            a, a, b, b, 0
        )[0] == 0.0 #js iterim test adjustment because `retrofit_hinge_loss` now returns a tuple of (hinge_loss, pre_clamp_hinge_loss)
    
    def test_retrofit_hinge_equal_dist(self):
        """if positive pair and negative pair are an equal distance 
        apart, loss should be equal to the margin"""
        a = torch.tensor([[1, 0, 1, 0]], dtype=float)
        b = torch.tensor([[0, 1, 0, 1]], dtype=float)
        gamma = 5.0
        assert self.experiment.retrofit_hinge_loss(
            a, b, a, b, gamma
        )[0] == gamma #js iterim test adjustment because `retrofit_hinge_loss` now returns a tuple of (hinge_loss, pre_clamp_hinge_loss)
    
    def test_retrofit_hinge_best_case_big_margin(self):
        """If positive pair are really close together, and negative
        pair are really far apart, that is good. So the loss should be
        zero (since the pairs are farther apart than the margin).
        """
         # a2 is a slight perturbation of a1
        a1 = torch.tensor([[1, 0, 1, 0]], dtype=float)
        a2 = a1 + torch.rand_like(a1) / 100.0
        # b2 is really far away from b1
        b1 = torch.tensor([[0, 100, 0, 100]], dtype=float)
        b2 = torch.tensor([[0, -100, 0, -100]], dtype=float) 
        gamma = 20.0
        assert self.experiment.retrofit_hinge_loss(
            a1, a2, b1, b2, gamma
        )[0] == 0.0 #js iterim test adjustment because `retrofit_hinge_loss` now returns a tuple of (hinge_loss, pre_clamp_hinge_loss)
    
    def test_retrofit_hinge_worst_case(self):
        """If positive pair are really far apart, and negative
        pair are really close together, that is bad. So the loss should be
        (much) greater than the margin
        """
        # a2 is really far away from a1
        a1 = torch.tensor([[0, 100, 0, 100]], dtype=float)
        a2 = torch.tensor([[0, -100, 0, -100]], dtype=float) 
        # b2 is a slight perturbation of b1
        b1 = torch.tensor([[1, 0, 1, 0]], dtype=float)
        b2 = a1 + torch.rand_like(a1) / 100.0
        gamma = 10.0
        loss = self.experiment.retrofit_hinge_loss(
            a1, a2, b1, b2, gamma
        )[0] #js iterim test adjustment because `retrofit_hinge_loss` now returns a tuple of (hinge_loss, pre_clamp_hinge_loss)
        assert loss > gamma
    
    def test_retrofit_hinge_clamp_batch(self):
        """The hinge loss should be applied along the batch dimension, to each example individually,
        so samples can't cancel each other out in the calculations.

        In this case, the loss tensor will be something like [145, -135]. So if we clamp, then average,
        we'll get an average of around 70. But if we average, and then clamp (which is wrong) then
        we'll get an average of around 5.
        """
        # a2 is really far away from a1
        a1 = torch.tensor([[0, 100, 0, 100], [0, 100, 0, 100]], dtype=float)
        a2 = torch.tensor([[0, -100, 0, -100], [0, 100, 0, 100]], dtype=float) 
        # b2 is a slight perturbation of b1
        b1 = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=float)
        b2 = a1 + torch.rand_like(a1) / 100.0
        gamma = 5.0
        loss = self.experiment.retrofit_hinge_loss(
            a1, a2, b1, b2, gamma
        )[0] #js iterim test adjustment because `retrofit_hinge_loss` now returns a tuple of (hinge_loss, pre_clamp_hinge_loss)
        assert loss > 50