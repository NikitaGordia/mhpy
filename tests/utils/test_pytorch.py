from unittest.mock import patch

import torch.nn as nn

from mhpy.utils.pytorch import get_model_size


class TestGetModelSize:
    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_simple_model(self, mock_logger):
        # Parameters: 10 * 5 (weights) + 5 (bias) = 55 parameters
        model = nn.Linear(10, 5)

        param_count = get_model_size(model)

        assert param_count == 55

    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_sequential_model(self, mock_logger):
        """Test get_model_size with a sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220
            nn.ReLU(),
            nn.Linear(20, 5),  # 20*5 + 5 = 105
        )

        param_count = get_model_size(model)

        assert param_count == 325

    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_conv_model(self, mock_logger):
        model = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        param_count = get_model_size(model)

        assert param_count == 448

    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_model_with_buffers(self, mock_logger):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
        )

        param_count = get_model_size(model)

        # Linear: 10*10 + 10 = 110
        # BatchNorm: 10 (weight) + 10 (bias) = 20
        # Total parameters: 130
        assert param_count == 130

    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_empty_model(self, mock_logger):
        model = nn.Sequential()

        param_count = get_model_size(model)

        assert param_count == 0

    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_custom_model(self, mock_logger):
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(5, 10)
                self.fc2 = nn.Linear(10, 2)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = CustomModel()
        param_count = get_model_size(model)

        # fc1: 5*10 + 10 = 60
        # fc2: 10*2 + 2 = 22
        # Total: 82
        assert param_count == 82

    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_different_dtypes(self, mock_logger):
        model = nn.Linear(10, 5).half()

        param_count = get_model_size(model)

        assert param_count == 55

    @patch("mhpy.utils.pytorch.logger")
    def test_get_model_size_large_model(self, mock_logger):
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
        )

        param_count = get_model_size(model)

        # Layer 1: 1000*500 + 500 = 500,500
        # Layer 2: 500*250 + 250 = 125,250
        # Layer 3: 250*10 + 10 = 2,510
        # Total: 628,260
        assert param_count == 628_260
