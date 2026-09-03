"""
Training Script for 2-Stage Acceleration & Kinematic Motion System:
Stage 1: Rest vs. Moving Motion Classification (MLP Zero-Velocity Detector)
Stage 2: Forward & Lateral Acceleration / Delta-Velocity Regression (Transformer)
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import prepare_acceleration_dataset
from models import RestMovingClassifierMLP, IMUTransformerTLIO

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def train_motion_classifier(cls_data: dict, epochs: int = 15, batch_size: int = 64, lr: float = 0.003) -> RestMovingClassifierMLP:
    print("\n==========================================================")
    print("   STAGE 1: TRAINING REST VS. MOVING CLASSIFIER (MLP)     ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classifier] Training on device: {device}")
    
    train_X = torch.tensor(cls_data['train_X'], dtype=torch.float32)
    train_y = torch.tensor(cls_data['train_y'], dtype=torch.long)
    val_X = torch.tensor(cls_data['val_X'], dtype=torch.float32).to(device)
    val_y = torch.tensor(cls_data['val_y'], dtype=torch.long).to(device)
    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    
    model = RestMovingClassifierMLP(input_dim=6, window_size=10, hidden_dim=64).to(device)
    
    n_rest = float(torch.sum(train_y == 0))
    n_moving = float(torch.sum(train_y == 1))
    weight_rest = len(train_y) / (2.0 * max(1.0, n_rest))
    weight_moving = len(train_y) / (2.0 * max(1.0, n_moving))
    class_weights = torch.tensor([weight_rest, weight_moving], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_acc = 0.0
    best_weights = None
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
            
        train_loss = epoch_loss / len(train_X)
        
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y).item()
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (torch.sum(val_preds == val_y).item() / len(val_y)) * 100.0
            
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[Classifier] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.2f}% (Best: {best_acc:.2f}%)")
            
    print(f"\n[Classifier] Finished training in {time.time() - start_time:.2f}s. Best Accuracy: {best_acc:.2f}%")
    
    model.load_state_dict(best_weights)
    model.eval()
    
    with torch.no_grad():
        val_preds = torch.argmax(model(val_X), dim=1).cpu().numpy()
        val_true = val_y.cpu().numpy()
        
        tp = np.sum((val_preds == 1) & (val_true == 1))
        tn = np.sum((val_preds == 0) & (val_true == 0))
        fp = np.sum((val_preds == 1) & (val_true == 0))
        fn = np.sum((val_preds == 0) & (val_true == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        print("----------------------------------------------------------")
        print("      STAGE 1: REST VS MOVING CLASSIFICATION METRICS      ")
        print("----------------------------------------------------------")
        print(f"Overall Accuracy:                   {best_acc:.2f}%")
        print(f"Moving Precision:                   {precision * 100:.2f}%")
        print(f"Moving Recall (Sensitivity):        {recall * 100:.2f}%")
        print(f"F1 Score:                           {f1 * 100:.2f}%")
        print(f"Confusion Matrix: [Rest Correct: {tn:,}, Moving Correct: {tp:,}, False Alarm: {fp:,}, Miss: {fn:,}]")
        print("----------------------------------------------------------")
        
    cls_pt = os.path.join(MODEL_DIR, "motion_classifier.pt")
    torch.save(model.state_dict(), cls_pt)
    print(f"[Classifier] Saved PyTorch model to {cls_pt}")
    
    return model


def train_acceleration_transformer(accel_data: dict, target_norm, epochs: int = 20, batch_size: int = 64, lr: float = 0.003) -> IMUTransformerTLIO:
    print("\n==========================================================")
    print("   STAGE 2: TRAINING ACCELERATION TRANSFORMER (Δv)       ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Transformer] Training on device: {device}")
    
    train_X = torch.tensor(accel_data['train_X'], dtype=torch.float32)
    train_y = torch.tensor(accel_data['train_y'], dtype=torch.float32)
    val_X = torch.tensor(accel_data['val_X'], dtype=torch.float32).to(device)
    val_y = torch.tensor(accel_data['val_y'], dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    
    model = IMUTransformerTLIO(
        input_dim=6,
        window_size=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_dim=2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
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
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
            
        train_loss = epoch_loss / len(train_X)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X)
            val_loss = criterion(val_preds, val_y).item()
            
        scheduler.step()
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[Transformer] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Best Val: {best_loss:.5f}")
            
    print(f"\n[Transformer] Finished training in {time.time() - start_time:.2f}s. Best Val Loss: {best_loss:.5f}")
    
    model.load_state_dict(best_weights)
    model.eval()
    
    with torch.no_grad():
        val_preds_norm = model(val_X).cpu().numpy()
        val_true_norm = val_y.cpu().numpy()
        
        val_preds_phys = target_norm.inverse_transform(val_preds_norm)
        val_true_phys = target_norm.inverse_transform(val_true_norm)
        
        fwd_true_dv = val_true_phys[:, 1]
        fwd_pred_dv = val_preds_phys[:, 1]
        
        mae_dv_mps2 = np.mean(np.abs(fwd_pred_dv - fwd_true_dv))
        mae_dv_kmh_s = mae_dv_mps2 * 3.6
        
        ss_res = np.sum((fwd_true_dv - fwd_pred_dv) ** 2)
        ss_tot = np.sum((fwd_true_dv - np.mean(fwd_true_dv)) ** 2)
        r2_score = 1.0 - (ss_res / (ss_tot + 1e-8))
        
        print("----------------------------------------------------------")
        print("   STAGE 2: ACCELERATION ESTIMATION METRICS (Δv)          ")
        print("----------------------------------------------------------")
        print(f"Acceleration Estimation Error (MAE): {mae_dv_mps2:.4f} m/s² ({mae_dv_kmh_s:.2f} km/h per second)")
        print(f"R² Score of Acceleration Delta (Δv):  {r2_score * 100:.2f}%")
        print("----------------------------------------------------------")
        
    trans_pt = os.path.join(MODEL_DIR, "tlio_transformer.pt")
    torch.save(model.state_dict(), trans_pt)
    print(f"[Transformer] Saved PyTorch model to {trans_pt}")
    
    return model


if __name__ == "__main__":
    print("Loading and preparing IO-VNBD dataset for 2-Stage Acceleration System...")
    data_pkg, feat_norm, target_norm = prepare_acceleration_dataset(["S-S1", "S-S2"], window_size=10, stride=2)
    
    # 1. Train Stage 1 Motion Classifier (MLP)
    cls_model = train_motion_classifier(data_pkg['cls'], epochs=15)
    
    # 2. Train Stage 2 Acceleration Transformer
    transformer_model = train_acceleration_transformer(data_pkg['accel'], target_norm, epochs=20)
    
    print("\n2-Stage Acceleration & Kinematic System training completed successfully!")
