"""
Neural Network Architectures for IMU-Based Motion & Vector Tracking
Includes:
1. SimpleRNN: Recurrent neural network for sequential IMU data with explicit hidden state.
2. SimpleMLP: Multi-layer perceptron for instantaneous frame-by-frame sensor mapping.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any


class SimpleRNNCell(nn.Module):
    """
    Standard Elman RNN Cell where:
    h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
    y_t = W_ho @ h_t + b_ho (2D vector / velocity / heading)
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 32, output_size: int = 5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc_ih = nn.Linear(input_size, hidden_size)
        self.fc_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.act = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step forward:
        x_t: (batch, input_size)
        h_prev: (batch, hidden_size)
        Returns: (y_t, h_t)
        """
        h_t = self.act(self.fc_ih(x_t) + self.fc_hh(h_prev))
        y_t = self.fc_out(h_t)
        return y_t, h_t


class SimpleRNN(nn.Module):
    """
    Sequential RNN wrapper supporting both sequence batch processing
    and real-time streaming single-step inference.
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 32, output_size: int = 5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = SimpleRNNCell(input_size, hidden_size, output_size)

    def forward(self, x_seq: torch.Tensor, h_0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: (batch, seq_len, input_size)
        h_0: (batch, hidden_size)
        Returns:
            outputs: (batch, seq_len, output_size)
            h_final: (batch, hidden_size)
        """
        batch_size, seq_len, _ = x_seq.shape
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x_seq.device)
            
        h_t = h_0
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            y_t, h_t = self.cell(x_t, h_t)
            outputs.append(y_t.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        return outputs, h_t

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single millisecond/sample step for live streaming."""
        if h_prev is None:
            batch_size = x_t.shape[0] if x_t.dim() > 1 else 1
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x_t.device)
        return self.cell(x_t, h_prev)


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron for non-sequential IMU data mapping:
    Maps [ax, ay, az, gx, gy, gz] -> [vx, vy, speed, dir_x, dir_y]
    """
    def __init__(self, input_size: int = 6, hidden_size: int = 64, output_size: int = 5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_size) or (batch, seq_len, input_size)
        Returns: (batch, output_size)
        """
        return self.net(x)


def get_model(model_type: str = "rnn", input_size: int = 6, output_size: int = 5) -> nn.Module:
    if model_type.lower() == "rnn":
        return SimpleRNN(input_size=input_size, hidden_size=32, output_size=output_size)
    elif model_type.lower() == "mlp":
        return SimpleMLP(input_size=input_size, hidden_size=64, output_size=output_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'rnn' or 'mlp'.")
