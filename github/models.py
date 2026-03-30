import math
import torch
import torch.nn as nn

# Multi-task loss with task-dependent homoscedastic uncertainty-based weighting
class MultiTaskLossWithUncertainty(nn.Module):

    def __init__(self, num_tasks=2):
        super(MultiTaskLossWithUncertainty, self).__init__()
        
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        assert len(losses) == len(self.log_vars), \
            f"Number of losses ({len(losses)}) must match number of tasks ({len(self.log_vars)})"
        
        total_loss = 0
        weights = []
        
        for loss, log_var in zip(losses, self.log_vars):
            
            precision = torch.exp(-log_var)
            weighted_loss = precision * loss + log_var / 2.0
            
            total_loss += weighted_loss
            weights.append(precision.item())
        
        return total_loss, torch.tensor(weights)
    
    def get_weights(self):
        return torch.exp(-self.log_vars).detach().cpu().numpy()
    
    def get_sigmas(self):
        return torch.exp(self.log_vars / 2.0).detach().cpu().numpy()

# Task-specific attention block based on the Multi-Task Attention Network (MTAN)
class TaskAttentionBlock(nn.Module):
    
    def __init__(self, d_model, first_block, hidden_ratio=1.0, dropout=0.0):
        super().__init__()
        
        self.first_block = first_block
        in_dim = d_model if first_block else 2 * d_model
        hidden_dim = max(1, int(d_model * hidden_ratio))
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, u, a_prev=None):
        # First MTAN block uses only the shared representation;
        # later blocks use both the current shared representation and previous task-specific representation.
        if self.first_block:
            x = u
        else:
            if a_prev is None:
                raise ValueError("a_prev required for non-first block.")
            x = torch.cat([u, a_prev], dim=-1)
            
        mask = torch.sigmoid(self.fc2(self.drop(self.act(self.fc1(x)))))
        
        return mask * u, mask

# Ocean Color 3 MODIS algorithm (OC3M) calculation module using predicted Rrs
class OC3Calculator(nn.Module):

    def __init__(self, mid_vars):
        super(OC3Calculator, self).__init__()
        
        self.mid_vars = mid_vars
        self.available_steps = self._check_available_steps()
        self.num_available_steps = len(self.available_steps)
        
        # OC3M coefficients
        self.a0 = 0.26294
        self.a1 = -2.64669
        self.a2 = 1.28364
        self.a3 = 1.08209
        self.a4 = -1.76828
    
    def _check_available_steps(self):
        # Flexible OC3M calculation depending on available intermediate variables
        available = []
        
        if all(v in self.mid_vars for v in ['Rrs_443', 'Rrs_488', 'Rrs_547']):
            available.append(0)
        
        if all(v in self.mid_vars for v in ['Rrs_blue', 'Rrs_547']):
            available.append(1)
        
        if 'ratio' in self.mid_vars:
            available.append(2)
            
        if 'R' in self.mid_vars:
            available.append(3)

        if all(v in self.mid_vars for v in ['R', 'R2', 'R3', 'R4']):
            available.append(4)
        
        return available
    
    def _calculate_step(self, step_idx, rrs_denorm):

        if step_idx == 0:
            # Step 1: max(Rrs_443, Rrs_488) / Rrs_547
            idx_443 = self.mid_vars.index('Rrs_443')
            idx_488 = self.mid_vars.index('Rrs_488')
            idx_547 = self.mid_vars.index('Rrs_547')
            
            Rrs_443 = torch.clamp(rrs_denorm[:, idx_443:idx_443+1], min=1e-6)
            Rrs_488 = torch.clamp(rrs_denorm[:, idx_488:idx_488+1], min=1e-6)
            Rrs_547 = torch.clamp(rrs_denorm[:, idx_547:idx_547+1], min=1e-6)
            
            Rrs_blue = torch.maximum(Rrs_443, Rrs_488)
            ratio = Rrs_blue / (Rrs_547 + 1e-8)
            R = torch.log10(torch.clamp(ratio + 0.0001, min=1e-6))
            
        elif step_idx == 1:
            # Step 2: Rrs_blue / Rrs_547
            idx_blue = self.mid_vars.index('Rrs_blue')
            idx_547 = self.mid_vars.index('Rrs_547')
            
            Rrs_blue = torch.clamp(rrs_denorm[:, idx_blue:idx_blue+1], min=1e-6)
            Rrs_547 = torch.clamp(rrs_denorm[:, idx_547:idx_547+1], min=1e-6)
            
            ratio = Rrs_blue / (Rrs_547 + 1e-8)
            R = torch.log10(torch.clamp(ratio + 0.0001, min=1e-6))
            
        elif step_idx == 2:
            # Step 3: ratio
            idx_ratio = self.mid_vars.index('ratio')
            ratio = torch.clamp(rrs_denorm[:, idx_ratio:idx_ratio+1], min=1e-6)
            R = torch.log10(torch.clamp(ratio + 0.0001, min=1e-6))
            
        elif step_idx == 3:
            # Step 4: R
            idx_R = self.mid_vars.index('R')
            R = rrs_denorm[:, idx_R:idx_R+1]
            
        elif step_idx == 4:
            # Step 5: R, R2, R3, R4
            idx_R = self.mid_vars.index('R')
            idx_R2 = self.mid_vars.index('R2')
            idx_R3 = self.mid_vars.index('R3')
            idx_R4 = self.mid_vars.index('R4')
            
            R = rrs_denorm[:, idx_R:idx_R+1]
            R2 = rrs_denorm[:, idx_R2:idx_R2+1]
            R3 = rrs_denorm[:, idx_R3:idx_R3+1]
            R4 = rrs_denorm[:, idx_R4:idx_R4+1]
            
            log10_chla = self.a0 + self.a1*R + self.a2*R2 + self.a3*R3 + self.a4*R4
            return torch.clamp(10**log10_chla, min=0.0, max=500.0)
        
        if step_idx < 4:
            log10_chla = self.a0 + self.a1*R + self.a2*R**2 + self.a3*R**3 + self.a4*R**4
            return torch.clamp(10**log10_chla, min=0.0, max=500.0)
    
    def forward(self, rrs_denorm):
        
        chl_steps = []
        
        for step_idx in self.available_steps:
            chl_step = self._calculate_step(step_idx, rrs_denorm)
            chl_steps.append(chl_step)
        
        return torch.cat(chl_steps, dim=1)

# Positional encoding for SR band information
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer-based hierarchical multi-task model for Rrs and Chl-a prediction
class Transformer_OC_MTAN(nn.Module):
    
    def __init__(self, columns_dict,
                 d_model=64,
                 nhead=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 num_encoder_layers=3,
                 scalers=None,
                 mtan_hidden_ratio=1.0):
        super().__init__()

        self.d_model = d_model
        self.sr_input_dim = 1
        self.aux_input_dim = len(columns_dict['aux_input'])
        self.rrs_output_dim = len(columns_dict['mid_vars'])
        self.mid_vars = columns_dict['mid_vars']

        # Rrs scaler for inverse min-max scaling before OC3M calculation
        if scalers is not None and 'mid_vars' in scalers:
            rrs_scaler = scalers['mid_vars']
            self.register_buffer('rrs_min', torch.tensor(rrs_scaler.data_min_, dtype=torch.float32))
            self.register_buffer('rrs_max', torch.tensor(rrs_scaler.data_max_, dtype=torch.float32))
        else:
            self.register_buffer('rrs_min', None)
            self.register_buffer('rrs_max', None)

        # Transformer-based encoder for SR band information
        self.sr_projection = nn.Linear(self.sr_input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )
        self.sr_norm = nn.LayerNorm(self.d_model)

        # Self-attention pooling for spectral latent representation
        self.attention_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1)
        )

        # Auxiliary encoder for SR-derived Rrs, seasonal variables, and spectral indices
        self.aux_mlp = nn.Sequential(
            nn.Linear(self.aux_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model)
        )
        self.aux_norm = nn.LayerNorm(self.d_model)

        # Cross-attention between SR and auxiliary representations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.d_model)

        # Feature fusion for integrated latent representation
        self.fused_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Rrs intermediate prediction head
        self.rrs_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.rrs_output_dim)
        )
        self.rrs_projection = nn.Sequential(
            nn.Linear(self.rrs_output_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # OC3M-based Chl-a calculation and embedding
        self.oc3_calculator = OC3Calculator(columns_dict['mid_vars'])
        self.chl_oc3_projection = nn.Sequential(
            nn.Linear(self.oc3_calculator.num_available_steps, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Stage-wise task-specific attention blocks for Rrs and Chl-a tasks
        self.mtan_rrs = nn.ModuleList([
            TaskAttentionBlock(self.d_model, first_block=True,  hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
        ])
        self.mtan_chl = nn.ModuleList([
            TaskAttentionBlock(self.d_model, first_block=True,  hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
        ])

        # Final Chl-a prediction head
        self.chl_mlp = nn.Sequential(
            nn.Linear(self.d_model * 6, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def set_rrs_scaler(self, rrs_min, rrs_max):
        device = next(self.parameters()).device
        self.register_buffer('rrs_min', torch.tensor(rrs_min, dtype=torch.float32, device=device))
        self.register_buffer('rrs_max', torch.tensor(rrs_max, dtype=torch.float32, device=device))

    def _denormalize_rrs(self, rrs_normalized):
        if self.rrs_min is None or self.rrs_max is None:
            return rrs_normalized
        rrs_min = self.rrs_min.to(rrs_normalized.device)
        rrs_max = self.rrs_max.to(rrs_normalized.device)
        return rrs_normalized * (rrs_max - rrs_min) + rrs_min

    def _attention_pooling(self, x):
        # Attention pooling for spectral latent representation
        w = self.attention_net(x)
        w = torch.softmax(w, dim=1)
        pooled = torch.sum(x * w, dim=1)
        return pooled, w

    def _build_shared(self, sr_seq, aux_vars, return_attention=False):
        # Shared backbone: transformer-based encoder + auxiliary encoder
        attn = {}

        sr_emb = self.sr_projection(sr_seq)
        sr_emb = self.pos_encoder(sr_emb)
        sr_out = self.transformer_encoder(sr_emb)
        sr_out = self.sr_norm(sr_out + sr_emb)

        sr_hidden, pool_w = self._attention_pooling(sr_out)
        if return_attention:
            attn['pooling_weights'] = pool_w

        aux_hidden = self.aux_norm(self.aux_mlp(aux_vars))
        
        # Cross-attention from spectral to auxiliary representation
        q = sr_hidden.unsqueeze(1)
        kv = aux_hidden.unsqueeze(1)
        sr_attended, cross_w = self.cross_attention(query=q, key=kv, value=kv, need_weights=return_attention)
        sr_enhanced = self.cross_norm(sr_hidden + sr_attended.squeeze(1))

        if return_attention:
            attn['cross_attention'] = cross_w

        # Shared latent representations used in later task-specific pathways
        u1 = sr_enhanced
        u2 = self.fused_mlp(torch.cat([sr_hidden, aux_hidden], dim=1))

        return u1, u2, aux_hidden, attn

    def forward(self, sr_seq, aux_vars, return_attention=False):
        attention_dict = {}

        # 1) Shared latent representations from spectral and auxiliary inputs
        u1, u2, aux_hidden, attn_shared = self._build_shared(sr_seq, aux_vars, return_attention=return_attention)
        if return_attention:
            attention_dict.update(attn_shared)

        # 2) Stage-wise task-specific latent representations (MTAN)
        a_rrs0, mask_rrs0 = self.mtan_rrs[0](u1, None)
        a_chl0, mask_chl0 = self.mtan_chl[0](u1, None)

        a_rrs1, mask_rrs1 = self.mtan_rrs[1](u2, a_rrs0)
        a_chl1, mask_chl1 = self.mtan_chl[1](u2, a_chl0)

        # 3) Intermediate prediction of Rrs
        rrs_head_input = torch.cat([a_rrs0, aux_hidden], dim=1)  # [B, 2*d_model]
        rrs_out = self.rrs_mlp(rrs_head_input)

        # 4) OC3M-based Chl-a and Rrs-derived latent representations
        rrs_denorm = self._denormalize_rrs(rrs_out)
        chl_oc3_concat = self.oc3_calculator(rrs_denorm)
        chl_oc3_emb = self.chl_oc3_projection(chl_oc3_concat)    # u4
        rrs_embedded = self.rrs_projection(rrs_out)              # u3

        u3 = rrs_embedded
        u4 = chl_oc3_emb

        # 5) Additional MTAN stages using Rrs- and OC3M-derived representations
        a_rrs2, mask_rrs2 = self.mtan_rrs[2](u3, a_rrs1)
        a_chl2, mask_chl2 = self.mtan_chl[2](u3, a_chl1)

        a_rrs3, mask_rrs3 = self.mtan_rrs[3](u4, a_rrs2)
        a_chl3, mask_chl3 = self.mtan_chl[3](u4, a_chl2)

        # 6) Final Chl-a prediction using task-specific, auxiliary, Rrs, and OC3M pathways
        chl_input = torch.cat([
            a_chl0, a_chl2, a_chl3,
            aux_hidden, rrs_embedded, chl_oc3_emb
        ], dim=1)

        chl_out = self.chl_mlp(chl_input)

        if return_attention:
            attention_dict['chl_oc3_concat'] = chl_oc3_concat
            attention_dict['mtan_masks'] = {
                'rrs': [mask_rrs0, mask_rrs1, mask_rrs2, mask_rrs3],
                'chl': [mask_chl0, mask_chl1, mask_chl2, mask_chl3],
            }
            attention_dict['mtan_a'] = {
                'rrs': [a_rrs0, a_rrs1, a_rrs2, a_rrs3],
                'chl': [a_chl0, a_chl1, a_chl2, a_chl3],
            }
            return rrs_out, chl_out, attention_dict

        return rrs_out, chl_out