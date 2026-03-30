import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

from functions import set_seed, set_device, create_dataloader, train_model
from models import MultiTaskLossWithUncertainty, Transformer_OC_MTAN


set_seed()
device = set_device()

# Load 4.6 km resolution dataset
model_data = pd.read_parquet("/PATH/TO/YOUR/PARQUET/FILE/4.6 km resolution dataset.parquet")

SR_vars = ["SR_469", "SR_555", "SR_645", "SR_859", "SR_1240", "SR_1640", "SR_2130"]
aux_input = [
    "SR_469_Rrs", "SR_555_Rrs", "SR_645_Rrs", "SR_859_Rrs", "SR_1240_Rrs", "SR_1640_Rrs", "SR_2130_Rrs",
    "Ratio_blue","Ratio_blue_SR_Rrs","FAI","FAI_SR_Rrs","NDVI","NDVI_SR_Rrs",
    "DOY_sin", "DOY_cos", "season_sin", "season_cos"
    ]
mid_vars = ["Rrs_443","Rrs_469","Rrs_488","Rrs_547","Rrs_555","Rrs_645", "R"]
target_chl = ['Chl-a']

columns_dict = {'SR_vars':SR_vars, 'aux_input':aux_input, 'mid_vars':mid_vars, 'Chl_a':target_chl}

# Hyperparameter settings
EPOCHS = 500

batch_sizes = []
models_depths = []
num_attention_heads = []
num_encoder_layers = []
learning_rates = []
dropout_rates = []

# Directory for saving trained models
save_dir = "/PATH/TO/YOUR/SAVE/DIRECTORY"
os.makedirs(save_dir, exist_ok=True)

# Model training loop
for bs in batch_sizes:
    train_loader, vali_loader, test_loader, scalers = create_dataloader(model_data, columns_dict, bs)
    for md in models_depths:
        for nh in num_attention_heads:
            for nl in num_encoder_layers:
                for lr in learning_rates:
                    for dr in dropout_rates:
                        
                        fd = md*4
                        model = Transformer_OC_MTAN(
                            columns_dict=columns_dict,
                            d_model=md,
                            nhead=nh,
                            dim_feedforward=fd,
                            dropout=dr,
                            num_encoder_layers=nl,
                            scalers=scalers
                            )

                        # Multi-task loss
                        criterion = nn.MSELoss()
                        loss_fn = MultiTaskLossWithUncertainty(num_tasks=2)
                        
                        # Optimizer includes model and uncertainty parameters
                        optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

                        trained_model, trained_loss_fn = train_model(
                            model, train_loader, vali_loader, 
                            criterion, loss_fn, optimizer, scheduler, 
                            EPOCHS, device, print_on=True
                        )
                        
                        save_path = os.path.join(save_path , f"bs{bs}_md{md}_nh{nh}_nl{nl}_lr{lr}_dr{dr}.pth")
                        torch.save({
                            'model_state_dict': trained_model.state_dict(),
                            'loss_fn_state_dict': trained_loss_fn.state_dict(),
                            'learned_weights': trained_loss_fn.get_weights(),
                            'learned_sigmas': trained_loss_fn.get_sigmas(),
                            'hyperparams': {'bs': bs, 'md': md, 'nh': nh, 'nl': nl, 'lr': lr, 'dr': dr}
                            }, save_path)