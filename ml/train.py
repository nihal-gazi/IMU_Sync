"""
Training Script for IMU-Sync RNN & MLP Models
Trains models on IO-VNBD smartphone accelerometer & gyroscope readings.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from dataset import prepare_datasets
from models import SimpleRNN, SimpleMLP

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def train_rnn(rnn_data: dict, epochs: int = 25, batch_size: int = 64, lr: float = 0.003) -> SimpleRNN:
    print("\n==========================================")
    print("       TRAINING SIMPLERNN (SEQUENTIAL)     ")
    print("==========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RNN] Training on device: {device}")
    
    train_X = torch.tensor(rnn_data['train_X'], dtype=torch.float32)
    train_y = torch.tensor(rnn_data['train_y'], dtype=torch.float32)
    val_X = torch.tensor(rnn_data['val_X'], dtype=torch.float32).to(device)
    val_y = torch.tensor(rnn_data['val_y'], dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    
    model = SimpleRNN(input_size=6, hidden_size=32, output_size=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_mse = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_loss = float('inf')
    best_weights = None
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds, _ = model(bx)
            
            # Combined MSE loss on velocity & direction targets
            loss = criterion_mse(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_loss += loss.item() * len(bx)
            
        train_loss = epoch_loss / len(train_X)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds, _ = model(val_X)
            val_loss = criterion_mse(val_preds, val_y).item()
            
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[RNN] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Best Val: {best_loss:.5f}")
            
    print(f"[RNN] Training finished in {time.time() - start_time:.2f}s. Best Val Loss: {best_loss:.5f}")
    
    model.load_state_dict(best_weights)
    model.eval()
    
    # Save model weights
    pt_path = os.path.join(MODEL_DIR, "rnn_model.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[RNN] Saved PyTorch model to {pt_path}")
    
    return model


def train_mlp(mlp_data: dict, epochs: int = 25, batch_size: int = 128, lr: float = 0.003) -> SimpleMLP:
    print("\n==========================================")
    print("      TRAINING SIMPLEMLP (NON-SEQUENTIAL)  ")
    print("==========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MLP] Training on device: {device}")
    
    train_X = torch.tensor(mlp_data['train_X'], dtype=torch.float32)
    train_y = torch.tensor(mlp_data['train_y'], dtype=torch.float32)
    val_X = torch.tensor(mlp_data['val_X'], dtype=torch.float32).to(device)
    val_y = torch.tensor(mlp_data['val_y'], dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    
    model = SimpleMLP(input_size=6, hidden_size=64, output_size=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion_mse = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_loss = float('inf')
    best_weights = None
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            
            loss = criterion_mse(preds, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(bx)
            
        train_loss = epoch_loss / len(train_X)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X)
            val_loss = criterion_mse(val_preds, val_y).item()
            
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[MLP] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Best Val: {best_loss:.5f}")
            
    print(f"[MLP] Training finished in {time.time() - start_time:.2f}s. Best Val Loss: {best_loss:.5f}")
    
    model.load_state_dict(best_weights)
    model.eval()
    
    pt_path = os.path.join(MODEL_DIR, "mlp_model.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[MLP] Saved PyTorch model to {pt_path}")
    
    return model


if __name__ == "__main__":
    print("Loading and preparing IO-VNBD dataset...")
    data_dict, feat_norm, target_norm = prepare_datasets(["S-S1"])
    
    rnn_model = train_rnn(data_dict['rnn'], epochs=20)
    mlp_model = train_mlp(data_dict['mlp'], epochs=20)
    
    print("\nTraining completed successfully for both RNN and MLP!")
