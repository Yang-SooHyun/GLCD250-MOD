from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

import os
import random
import numpy as np
import copy


# =========================
# Reproducibility settings
# =========================
def set_seed(seed_value=42):
    # Fix random seeds for reproducibility
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # Fix CUDA-related randomness
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # Fix Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Device setup
# =========================
def set_device():
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[set_device] CUDA available: {torch.cuda.is_available()}")
    print(f"[set_device] CUDA version : {torch.version.cuda}")
    print(f"[set_device] Using device : {device}")
    return device

# =========================
# DataLoader creation
# =========================
def create_dataloader(df, columns_dict, batch_size, test_size=0.2, valid_size=0.1):
    
    set_seed()
    
    # Split dataset into train, validation, and test sets
    df_train_vali, df_test = train_test_split(df, test_size=test_size)
    valid_size_adjusted = valid_size / (1 - test_size)
    df_train, df_vali = train_test_split(df_train_vali, test_size=valid_size_adjusted)

    scalers = {}
    tensor_dict = {'train': {}, 'vali': {}, 'test': {}}
    
    # Surface reflectance sequence used as ordered spectral input
    SR_sequence_cols  = ["SR_469", "SR_555", "SR_645", "SR_859", "SR_1240", "SR_1640", "SR_2130"]

    for key, cols in columns_dict.items():

        train_df = df_train[cols]
        vali_df  = df_vali[cols]
        test_df  = df_test[cols]

        # SR sequence is kept in its original scale
        if key == "SR_vars":
            train_arr = train_df[SR_sequence_cols].values.astype('float32')
            vali_arr  = vali_df[SR_sequence_cols].values.astype('float32')
            test_arr  = test_df[SR_sequence_cols].values.astype('float32')

            tensor_dict['train'][key] = torch.tensor(train_arr).unsqueeze(-1)
            tensor_dict['vali'][key]  = torch.tensor(vali_arr).unsqueeze(-1)
            tensor_dict['test'][key]  = torch.tensor(test_arr).unsqueeze(-1)
            continue
        
        else:
            train_data = train_df.values
            vali_data  = vali_df.values
            test_data  = test_df.values

            scaler = MinMaxScaler()
            train_data = scaler.fit_transform(train_data)
            vali_data  = scaler.transform(vali_data)
            test_data  = scaler.transform(test_data)
            scalers[key] = scaler
    
            tensor_dict['train'][key] = torch.tensor(train_data, dtype=torch.float32)
            tensor_dict['vali'][key]  = torch.tensor(vali_data, dtype=torch.float32)
            tensor_dict['test'][key]  = torch.tensor(test_data, dtype=torch.float32)

    # Create TensorDataset objects
    train_dataset = TensorDataset(*[tensor_dict['train'][k] for k in columns_dict.keys()])
    vali_dataset  = TensorDataset(*[tensor_dict['vali'][k] for k in columns_dict.keys()])
    test_dataset  = TensorDataset(*[tensor_dict['test'][k] for k in columns_dict.keys()])

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    vali_loader  = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, vali_loader, test_loader, scalers


# =========================
# Model training
# =========================
def train_model(model, train_dataloader, valid_dataloader, criterion, loss_fn, optimizer, scheduler, EPOCHS, device, patience=30, print_on=False):
    
    set_seed()
    
    model.to(device)
    loss_fn.to(device)
    
    best_validation_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_fn_wts = copy.deepcopy(loss_fn.state_dict())
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        
        # Training phase
        model.train()
        loss_fn.train()
            
        total_train_loss = 0
        total_train_loss_rrs = 0
        total_train_loss_chl = 0
        total_grad_norm = 0
        num_batches = 0
        
        for sr_seq, aux_vars, y_rrs, y_chl in train_dataloader:
            sr_seq, aux_vars, y_rrs, y_chl = sr_seq.to(device), aux_vars.to(device), y_rrs.to(device), y_chl.to(device)
            
            optimizer.zero_grad()
            
            out_rrs, out_chl = model(sr_seq, aux_vars)
            
            loss_rrs = criterion(out_rrs, y_rrs)
            loss_chl = criterion(out_chl, y_chl)
            
            # Uncertainty-based multi-task loss weighting
            loss, _ = loss_fn([loss_rrs, loss_chl])
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_loss_rrs += loss_rrs.item()
            total_train_loss_chl += loss_chl.item()
            total_grad_norm += grad_norm.item()
            num_batches += 1
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_loss_rrs = total_train_loss_rrs / len(train_dataloader)
        avg_train_loss_chl = total_train_loss_chl / len(train_dataloader)
        avg_grad_norm = total_grad_norm / num_batches

        # Validation phase
        model.eval()
        loss_fn.eval()
            
        total_val_loss = 0
        total_val_loss_rrs = 0
        total_val_loss_chl = 0
        
        with torch.no_grad():
            for sr_seq, aux_vars, y_rrs, y_chl in valid_dataloader:
                sr_seq, aux_vars, y_rrs, y_chl = sr_seq.to(device), aux_vars.to(device), y_rrs.to(device), y_chl.to(device)
                
                out_rrs, out_chl = model(sr_seq, aux_vars)
                
                loss_rrs = criterion(out_rrs, y_rrs)
                loss_chl = criterion(out_chl, y_chl)
                loss, _ = loss_fn([loss_rrs, loss_chl])
                
                total_val_loss += loss.item()
                total_val_loss_rrs += loss_rrs.item()
                total_val_loss_chl += loss_chl.item()
                
        avg_val_loss = total_val_loss / len(valid_dataloader)
        avg_val_loss_rrs = total_val_loss_rrs / len(valid_dataloader)
        avg_val_loss_chl = total_val_loss_chl / len(valid_dataloader)

        scheduler.step(avg_val_loss_chl)

        # Early Stopping
        if avg_val_loss_chl < best_validation_loss:
            best_validation_loss = avg_val_loss_chl
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss_fn_wts = copy.deepcopy(loss_fn.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if print_on:
                    print(f"Early stopping at epoch {epoch}, best Chl val loss {best_validation_loss:.4f}")
                    weights = loss_fn.get_weights()
                    sigmas = loss_fn.get_sigmas()
                    print(f"Final learned weights: Rrs={weights[0]:.3f}, Chl={weights[1]:.3f}")
                    print(f"Final uncertainties: σ_Rrs={sigmas[0]:.3f}, σ_Chl={sigmas[1]:.3f}")
                break

        # Training log
        current_lr = optimizer.param_groups[0]['lr']
        if print_on and (no_improve == 0 or epoch % 5 == 0):
            scale = 1e4
            weights = loss_fn.get_weights()
            sigmas = loss_fn.get_sigmas()
            print(f"Epoch {epoch:3}/{EPOCHS} | "
                  f"Train: {avg_train_loss:.4f} (Rrs:{avg_train_loss_rrs*scale:.4f} Chl:{avg_train_loss_chl*scale:.4f}) | "
                  f"Val: {avg_val_loss:.4f} (Rrs:{avg_val_loss_rrs*scale:.4f} Chl:{avg_val_loss_chl*scale:.4f}) | "
                  f"w:[{weights[0]:.2f},{weights[1]:.2f}] σ:[{sigmas[0]:.2f},{sigmas[1]:.2f}] | "
                  f"GradNorm:{avg_grad_norm:.2f} | LR:{current_lr:.6f} | no_improve:{no_improve}")
    
    # Restore best model weights
    model.load_state_dict(best_model_wts)
    loss_fn.load_state_dict(best_loss_fn_wts)
    
    model.to("cpu")
    loss_fn.to("cpu")
    
    return model, loss_fn