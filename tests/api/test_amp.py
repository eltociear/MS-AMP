# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for amp api in MS-AMP."""

import unittest
import torch

import msamp
from msamp.nn.linear import FP8Linear
from msamp.optim import LBAdamW


class AmpTestCase(unittest.TestCase):
    """Test amp api."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    def test_initialize(self):
        """Test initialize function."""
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters())

        model, optimizer = msamp.initialize(model, optimizer)

        self.assertTrue(isinstance(model, FP8Linear))
        self.assertTrue(isinstance(optimizer, LBAdamW))
