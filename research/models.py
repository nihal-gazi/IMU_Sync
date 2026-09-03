"""
Neural Network Architectures for IMU-Based Motion & TLIO
Includes:
1. RestMovingClassifierMLP: Fast MLP classifier for stationary (REST) vs active (MOVING) motion detection.
2. IMUTransformerTLIO: 1-second window Multi-Head Attention Transformer for local body-frame displacement.
3. SimpleRNN & SimpleMLP baselines.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class RestMovingClassifierMLP(nn.Module):
    """
    Stationary vs. Moving Motion Classifier (Stage 1 Gating Model):
    Input: (B, 10, 6) -> Flattened 1-second 6-axis IMU window (60 features)
    Output: (B, 2) -> Binary Classification Logits [logit_rest, logit_moving]
    """
    def __init__(self, input_dim: int = 6, window_size: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.flat_dim = input_dim * window_size # 60
        self.net = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # [0: REST, 1: MOVING]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 10, 6) or (B, 60)
        Returns:
            Logits of shape (B, 2)
        """
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


class IMUTransformerTLIO(nn.Module):
    """
    1-Second Window IMU-Transformer for Local Body-Frame Displacement:
    Input: (B, 10, 6) -> 1-second continuous 6-axis IMU window [ax, ay, az, gx, gy, gz]
    Output: (B, 2) -> Local body-frame displacement [dx_lateral, dy_forward]
    """
    def __init__(
        self,
        input_dim: int = 6,
        window_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 2
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        
        # 1. Feature Embedding Projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Learnable 1D Positional Encodings
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # 3. Multi-Head Attention Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        # 4. Displacement Regression Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :s, :]
        h = self.transformer_encoder(h)
        h = self.norm(h)
        h_pool = torch.mean(h, dim=1)
        displacement = self.head(h_pool)
        return displacement


class SimpleRNNCell(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 32, output_size: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc_ih = nn.Linear(input_size, hidden_size)
        self.fc_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.act = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_t = self.act(self.fc_ih(x_t) + self.fc_hh(h_prev))
        y_t = self.fc_out(h_t)
        return y_t, h_t


class SimpleRNN(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 32, output_size: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = SimpleRNNCell(input_size, hidden_size, output_size)

    def forward(self, x_seq: torch.Tensor, h_0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
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


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 64, output_size: int = 2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_dim if 'hidden_dim' in locals() else hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
