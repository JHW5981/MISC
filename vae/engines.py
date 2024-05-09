import torch
from utils import *
import os
import sys
import math

def train_one_epoch(model:torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    **kwargs):
    model.train()
    loss_lst = []
    recons_loss_lst = []
    kld_loss_lst = []
    for sample_batch in data_loader:
        sample_batch.to(kwargs["device"])

        results = model(sample_batch)
        train_stats = model.module.loss_function(results, **kwargs)
        loss = train_stats['loss']

        # reduce losses over all GPUs for logging purposes
        train_stats_reduced = reduce_dict(train_stats)
        loss_reduced = train_stats_reduced["loss"].item()
        loss_lst.append(loss_reduced)
        recons_loss_lst.append(train_stats_reduced["Reconstruction_loss"].item())
        kld_loss_lst.append(train_stats_reduced["KLD"].item())

        if not math.isfinite(loss_reduced):
            print("Loss is {}, stopping training".format(loss_reduced))
            print(train_stats_reduced)
            sys.exit(1)    

        optimizer.zero_grad()
        loss.backward()
        if kwargs['clip_max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs['clip_max_norm'])
        optimizer.step()

    return {'loss': sum(loss_lst)/len(loss_lst), "Reconstruction_loss": sum(recons_loss_lst)/len(recons_loss_lst), "KLD": sum(kld_loss_lst)/len(kld_loss_lst)}



