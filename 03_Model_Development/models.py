"""Released model architecture; parameter names retain checkpoint compatibility."""
import math
import torch
from torch import nn

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

class TaskAttentionBlock(nn.Module):
    """
    MTAN attention: j=1이면 mask = sigmoid(W2(relu(W1(u))));
    j>=2이면 mask = sigmoid(W2(relu(W1(concat(u, a_prev))))). a = mask * u (per-channel).
    """
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
        if self.first_block:
            x = u
        else:
            if a_prev is None:
                raise ValueError("a_prev required for non-first block.")
            x = torch.cat([u, a_prev], dim=-1)
        mask = torch.sigmoid(self.fc2(self.drop(self.act(self.fc1(x)))))
        return mask * u, mask

class OCxCalculator(nn.Module):

    DEFAULT_CONFIG = {
        'low_ratio_col': 'max_Rrs_412_Rrs_443_Rrs_488/Rrs_547',
        'high_ratio_col': 'max_Rrs_412_Rrs_443_Rrs_488/Rrs_547',
        'low_coefs': [-0.14471456898707660, -4.49295453250624810,
                      1.56446770620056608, -0.13256524097893285,
                      -0.28815879834898661],
        'high_coefs': [0.21383900650409210, -2.47587295479457481,
                       1.91209390524842782, 1.17778471481358893,
                       -1.62506612850538179],
        't1': 0.79779644838836417,
        't2': 3.73704908174786876,
        'transition_basis': 'low',
    }

    def __init__(self, mid_vars, config=None):
        super().__init__()
        self.mid_vars = mid_vars
        self.num_available_steps = 1
        config = {**self.DEFAULT_CONFIG, **(config or {})}
        if config.get('transition_basis', 'low') != 'low':
            raise ValueError("Only transition_basis='low' is supported")
        if len(config['low_coefs']) != 5 or len(config['high_coefs']) != 5:
            raise ValueError('Each OCx coefficient set must contain five values')
        self.low_ratio_col = config['low_ratio_col']
        self.high_ratio_col = config['high_ratio_col']
        self.low_coefs = [float(value) for value in config['low_coefs']]
        self.high_coefs = [float(value) for value in config['high_coefs']]
        self.t1, self.t2 = float(config['t1']), float(config['t2'])
        if not (0 < self.t1 < self.t2):
            raise ValueError('OCx thresholds must satisfy 0 < t1 < t2')

        required = sorted(set(self._component_names(self.low_ratio_col)
                              + self._component_names(self.high_ratio_col)))
        missing  = [v for v in required if v not in self.mid_vars]
        if missing: raise ValueError(f"OCxCalculator missing required variables: {missing}")

    @staticmethod
    def _rrs_extract(expr):
        parts = str(expr).split('_')
        return [f'Rrs_{parts[i + 1]}' for i, part in enumerate(parts[:-1]) if part == 'Rrs']

    @classmethod
    def _component_names(cls, ratio_col):
        if '/' not in ratio_col:
            raise ValueError(f'Invalid ratio name: {ratio_col}')
        numerator, denominator = ratio_col.split('/', 1)
        return cls._rrs_extract(numerator) + cls._rrs_extract(denominator)

    def _col(self, rrs_denorm, name):
        idx = self.mid_vars.index(name)
        return rrs_denorm[:, idx:idx+1]

    def _component(self, rrs_denorm, expr):
        columns = [torch.clamp(self._col(rrs_denorm, name), min=1e-6)
                   for name in self._rrs_extract(expr)]
        if not columns:
            raise ValueError(f'Cannot parse Rrs component: {expr}')
        values = torch.cat(columns, dim=1)
        if expr.startswith('max_'):
            return torch.max(values, dim=1, keepdim=True).values
        if expr.startswith('mean_'):
            return torch.mean(values, dim=1, keepdim=True)
        return columns[0]

    def _ratio(self, rrs_denorm, ratio_col):
        numerator, denominator = ratio_col.split('/', 1)
        return self._component(rrs_denorm, numerator) / self._component(rrs_denorm, denominator)

    def _ocx_chla(self, ratio, coefs):
        ratio = torch.clamp(ratio, min=1e-6)
        x = torch.log10(ratio)

        a0, a1, a2, a3, a4 = coefs
        log_chla = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4
        log_chla = torch.clamp(log_chla, min=-5.0, max=5.0)

        return 10 ** log_chla

    def forward(self, rrs_denorm):
        low_ratio = self._ratio(rrs_denorm, self.low_ratio_col)
        high_ratio = self._ratio(rrs_denorm, self.high_ratio_col)
        chl_low = self._ocx_chla(low_ratio, self.low_coefs)
        chl_high = self._ocx_chla(high_ratio, self.high_coefs)

        basis = chl_low
        use_low = basis < self.t1
        use_high = basis > self.t2
        use_blend = ~(use_low | use_high)

        weight_high = (basis - self.t1) / (self.t2 - self.t1)
        chl_blend = (1.0 - weight_high) * chl_low + weight_high * chl_high

        re_chla = torch.where(use_low, chl_low, chl_blend)
        re_chla = torch.where(use_high, chl_high, re_chla)
        re_chla = torch.where(use_blend, chl_blend, re_chla)

        return re_chla

class HierarchicalRrsChlaTransformer_ver2(nn.Module):

    def __init__(self, columns_dict,
                 d_model=64,
                 nhead=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 num_encoder_layers=3,
                 scalers=None,
                 mtan_hidden_ratio=1.0,
                 ocx_config=None):

        super().__init__()

        self.d_model = d_model

        self.seq_input_vars = columns_dict['seq_input']
        self.seq_input_dim = len(self.seq_input_vars)
        self.seq_len = len(self.seq_input_vars[0])

        assert all(len(group) == self.seq_len for group in self.seq_input_vars), \
            "All seq_input variable groups must have the same length."

        self.aux_input_dim = len(columns_dict['aux_input'])
        self.rrs_output_dim = len(columns_dict['mid_rrs'])
        self.mid_rrs_vars = columns_dict['mid_rrs']

        rrs_scaler = scalers['mid_rrs']
        self.register_buffer('rrs_min', torch.tensor(rrs_scaler.data_min_, dtype=torch.float32))
        self.register_buffer('rrs_max', torch.tensor(rrs_scaler.data_max_, dtype=torch.float32))

        # ---------- Shared backbone ----------
        self.sr_projection = nn.Linear(self.seq_input_dim, self.d_model)
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
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False
        )
        self.sr_norm = nn.LayerNorm(self.d_model)

        self.attention_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1)
        )

        self.aux_mlp = nn.Sequential(
            nn.Linear(self.aux_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model)
        )
        self.aux_norm = nn.LayerNorm(self.d_model)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.d_model)

        self.fused_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---------- Rrs prediction + OCx prior ----------
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

        self.ocx_calculator = OCxCalculator(self.mid_rrs_vars, config=ocx_config)
        self.ocx_residual_scale = 0.2
        self.target_is_log = any("log" in v.lower() for v in columns_dict['target'])

        target_scaler = scalers['target']
        self.register_buffer('target_min', torch.tensor(target_scaler.data_min_, dtype=torch.float32))
        self.register_buffer('target_max', torch.tensor(target_scaler.data_max_, dtype=torch.float32))

        self.mtan_rrs = nn.ModuleList([
            TaskAttentionBlock(self.d_model, first_block=True,  hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
        ])
        self.mtan_chl = nn.ModuleList([
            TaskAttentionBlock(self.d_model, first_block=True,  hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
            TaskAttentionBlock(self.d_model, first_block=False, hidden_ratio=mtan_hidden_ratio, dropout=dropout),
        ])

        # correction_input = [a_chl0, a_chl1, a_chl2, aux_hidden, rrs_embedded, log_ocx, scaled_ocx]
        self.chl_mlp = nn.Sequential(
            nn.Linear(self.d_model * 5 + 2, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

        self._init_weights()
        nn.init.zeros_(self.chl_mlp[-1].weight)
        nn.init.zeros_(self.chl_mlp[-1].bias)

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
        w = self.attention_net(x)
        w = torch.softmax(w, dim=1)
        pooled = torch.sum(x * w, dim=1)
        return pooled, w

    def _build_shared(self, seq_input, aux_vars, return_attention=False):
        attn = {}

        sr_emb = self.sr_projection(seq_input)
        sr_emb = self.pos_encoder(sr_emb)
        sr_out = self.transformer_encoder(sr_emb)
        sr_out = self.sr_norm(sr_out + sr_emb)

        sr_hidden, pool_w = self._attention_pooling(sr_out)
        if return_attention:
            attn['pooling_weights'] = pool_w

        aux_hidden = self.aux_norm(self.aux_mlp(aux_vars))

        q = sr_hidden.unsqueeze(1)
        kv = aux_hidden.unsqueeze(1)
        sr_attended, cross_w = self.cross_attention(query=q, key=kv, value=kv, need_weights=return_attention)
        sr_enhanced = self.cross_norm(sr_hidden + sr_attended.squeeze(1))

        if return_attention:
            attn['cross_attention'] = cross_w

        u1 = sr_enhanced
        u2 = self.fused_mlp(torch.cat([sr_hidden, aux_hidden], dim=1))

        return u1, u2, aux_hidden, attn

    def _transform_ocx_scalar(self, chl_ocx):
        return torch.log10(torch.clamp(chl_ocx, min=0.0) + 1.0)

    def _scale_ocx_to_target(self, chl_ocx):
        if self.target_is_log:
            chl_ocx = torch.log10(torch.clamp(chl_ocx, min=1e-6))

        target_min = self.target_min.to(chl_ocx.device)
        target_max = self.target_max.to(chl_ocx.device)
        return (chl_ocx - target_min) / (target_max - target_min + 1e-8)

    def forward(
        self,
        seq_input,
        aux_vars,
        return_attention=False,
        return_latents=False,
    ):
        attention_dict = {}

        # 1) Shared backbone -> u1,u2, aux_hidden
        u1, u2, aux_hidden, attn_shared = self._build_shared(seq_input, aux_vars, return_attention=return_attention)
        if return_attention:
            attention_dict.update(attn_shared)

        # 2) MTAN stage 1 on u1,u2
        a_rrs0, mask_rrs0 = self.mtan_rrs[0](u1, None)
        a_chl0, mask_chl0 = self.mtan_chl[0](u1, None)

        a_rrs1, mask_rrs1 = self.mtan_rrs[1](u2, a_rrs0)
        a_chl1, mask_chl1 = self.mtan_chl[1](u2, a_chl0)

        # 3) Rrs head: gated u1 + aux
        rrs_head_input = torch.cat([a_rrs0, aux_hidden], dim=1)
        rrs_out = self.rrs_mlp(rrs_head_input)

        # 4) Build Rrs embedding and scalar OCx prior
        rrs_denorm = self._denormalize_rrs(rrs_out)
        chl_ocx = self.ocx_calculator(rrs_denorm)
        log_chl_ocx = self._transform_ocx_scalar(chl_ocx)
        chl_ocx_scaled = self._scale_ocx_to_target(chl_ocx)
        rrs_embedded = self.rrs_projection(rrs_out)

        # 5) MTAN stage 2 on Rrs embedding
        a_rrs2, mask_rrs2 = self.mtan_rrs[2](rrs_embedded, a_rrs1)
        a_chl2, mask_chl2 = self.mtan_chl[2](rrs_embedded, a_chl1)

        # 6) Chl head: scaled OCx baseline + bounded neural residual correction
        correction_input = torch.cat([
            a_chl0, a_chl1, a_chl2,
            aux_hidden, rrs_embedded, log_chl_ocx, chl_ocx_scaled
        ], dim=1)

        correction = self.ocx_residual_scale * torch.tanh(self.chl_mlp(correction_input))
        chl_out = chl_ocx_scaled + correction

        if return_latents:
            latent_dict = {
                "chl_correction_input": correction_input,
            }
            if return_attention:
                attention_dict['chl_ocx'] = chl_ocx
                attention_dict['log_chl_ocx'] = log_chl_ocx
                attention_dict['chl_ocx_scaled'] = chl_ocx_scaled
                attention_dict['chl_ocx_correction'] = correction
                attention_dict['mtan_masks'] = {
                    'rrs': [mask_rrs0, mask_rrs1, mask_rrs2],
                    'chl': [mask_chl0, mask_chl1, mask_chl2],
                }
                attention_dict['mtan_a'] = {
                    'rrs': [a_rrs0, a_rrs1, a_rrs2],
                    'chl': [a_chl0, a_chl1, a_chl2],
                }
                return rrs_out, chl_out, attention_dict, latent_dict
            return rrs_out, chl_out, latent_dict

        if return_attention:
            attention_dict['chl_ocx'] = chl_ocx
            attention_dict['log_chl_ocx'] = log_chl_ocx
            attention_dict['chl_ocx_scaled'] = chl_ocx_scaled
            attention_dict['chl_ocx_correction'] = correction
            attention_dict['mtan_masks'] = {
                'rrs': [mask_rrs0, mask_rrs1, mask_rrs2],
                'chl': [mask_chl0, mask_chl1, mask_chl2],
            }
            attention_dict['mtan_a'] = {
                'rrs': [a_rrs0, a_rrs1, a_rrs2],
                'chl': [a_chl0, a_chl1, a_chl2],
            }
            return rrs_out, chl_out, attention_dict

        return rrs_out, chl_out

class MultiTaskLossWithUncertainty(nn.Module):
    """
    Multi-task loss weighting using homoscedastic uncertainty.
    
    For regression tasks:
        L = exp(-s) * L_task + s/2
    
    where s = log(σ²) is the learnable log variance.
    
    Reference:
        Kendall et al. (2018), "Multi-Task Learning Using Uncertainty to Weigh Losses"
    
    Args:
        num_tasks (int): Number of tasks (default: 2 for Rrs and Chl-a)
    """
    def __init__(self, num_tasks=2):
        super(MultiTaskLossWithUncertainty, self).__init__()
        
        # Initialize log variance parameters (s = log σ²)
        # Starting at 0.0 means σ² = 1 (neutral weighting)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses, task_indices=None):
        """
        Args:
            losses (list or tuple): List of task losses [loss_task1, loss_task2, ...]
            task_indices (list or tuple, optional): Indices of active tasks.
                For example, [0, 1] activates only the first two tasks of
                a three-task loss during staged warm-up.
        
        Returns:
            total_loss (torch.Tensor): Weighted multi-task loss
            weights (torch.Tensor): Effective weights for each task (for monitoring)
        """
        if task_indices is None:
            task_indices = list(range(len(self.log_vars)))
        if len(losses) != len(task_indices):
            raise ValueError(
                f"Number of losses ({len(losses)}) must match active task "
                f"indices ({len(task_indices)})."
            )
        if any(index < 0 or index >= len(self.log_vars) for index in task_indices):
            raise IndexError(
                f"Task indices {task_indices} are invalid for "
                f"{len(self.log_vars)} tasks."
            )
        
        total_loss = 0
        weights = []
        
        for loss, task_index in zip(losses, task_indices):
            log_var = self.log_vars[task_index]
            # Precision (inverse variance): exp(-s) = exp(-log(σ²)) = 1/σ²
            precision = torch.exp(-log_var)
            
            # Released implementation: precision * MSE + log(variance)/2.
            weighted_loss = precision * loss + log_var / 2.0
            
            total_loss += weighted_loss
            weights.append(precision.item())
        
        return total_loss, torch.tensor(weights)
    
    def get_weights(self):
        """
        Returns the current uncertainty-based weights (1/σ²) for each task.
        """
        return torch.exp(-self.log_vars).detach().cpu().numpy()
    
    def get_sigmas(self):
        """
        Returns the current standard deviations (σ) for each task.
        """
        return torch.exp(self.log_vars / 2.0).detach().cpu().numpy()
