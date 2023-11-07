from dataset import BraggNNDataset
import h5py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from model import NLB
import torch, argparse, os, time, sys, shutil, logging
from util import str2bool, str2tuple, s2ituple
from torch.utils.data import DataLoader
import numpy as np
import csv
from tqdm import tqdm
import optuna
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn




class BraggNN(nn.Module):
    def __init__(self, imgsz, cnn_channels=(64, 32, 8), fcsz=(64, 32, 16, 8)):
        super(BraggNN, self).__init__()
        self.cnn_ops = nn.ModuleList()
        cnn_in_chs = (1, ) + cnn_channels[:-1]

        fsz = imgsz
        for ic, oc in zip(cnn_in_chs, cnn_channels):
            self.cnn_ops.append(nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=0))
            self.cnn_ops.append(nn.LeakyReLU(negative_slope=0.01))
            fsz -= 2  # adjust the size due to convolution without padding

        self.nlb = NLB(in_ch=cnn_channels[0])

        self.dense_ops = nn.ModuleList()
        dense_in_chs = (fsz * fsz * cnn_channels[-1], ) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops.append(nn.Linear(ic, oc))
            self.dense_ops.append(nn.LeakyReLU(negative_slope=0.01))

        # Output layer
        self.dense_ops.append(nn.Linear(fcsz[-1], 2))

    def forward(self, x):
        _out = x
        for layer in self.cnn_ops[:1]:
            _out = layer(_out)

        _out = self.nlb(_out)

        for layer in self.cnn_ops[1:]:
            _out = layer(_out)

        # _out = _out.view(_out.size(0), -1)  # Flatten the tensor for the dense layer
        _out = _out.reshape(_out.size(0), -1)

        for layer in self.dense_ops:
            _out = layer(_out)

        return _out



def create_scheduler(optimizer, trial):
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ExponentialLR'])
    if scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 1, 100)
        gamma = trial.suggest_float('gamma', 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = trial.suggest_int('T_max', 1, 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == 'ExponentialLR':
        gamma = trial.suggest_float('gamma', 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return scheduler

def setup_data_loaders(batch_size):
    ds_train= BraggNNDataset(psz=IMG_SIZE, rnd_shift=aug, use='train')
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, drop_last=True, pin_memory=True)
    #TODO: Change prefetch_factor back to 2 and pin_memory to true

    ds_valid = BraggNNDataset(psz=IMG_SIZE, rnd_shift=0, use='validation')
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2, drop_last=False, pin_memory=True)

    return dl_train, dl_valid


batch_size_temp=256
IMG_SIZE = 11
FC_LAYER_SIZES = (64, 32, 16, 8)  # example sizes of the fully connected layers
aug=1
num_epochs=20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading data...')
train_loader, valid_loader = setup_data_loaders(batch_size_temp)
print('Data loaded')
# Takes 5 million years to run

IMG_SIZE = 11
FC_LAYER_SIZES = (64, 32, 16, 8)  # example sizes of the fully connected layers, should be found through global search
aug=1
num_epochs=100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-9, 1e-3, log=True)
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])

    # Initialize the model
    model = BraggNN(imgsz=IMG_SIZE, fcsz=FC_LAYER_SIZES).to(device)


    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 1.0)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)



    # Initialize the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # optimizer = create_optimizer(model, trial)

    # Optionally create a scheduler
    scheduler = create_scheduler(optimizer, trial)

    # Setup the data loaders
    # train_loader, valid_loader = setup_data_loaders(batch_size)

    # Define loss function
    criterion = torch.nn.MSELoss()

    # Training loop
    # for epoch in range(num_epochs):
    previous_epoch_loss = 100
    progress_bar = tqdm(range(num_epochs), disable=True)

    best_validation_loss = float('inf')
    checkpoint_dir = "./model_checkpoints"  # Directory where checkpoints will be saved
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists



    for epoch in progress_bar:
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                validation_loss += loss.item()

        validation_loss /= len(valid_loader)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"trial{trial.number}-epoch{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            trial.set_user_attr("checkpoint_path", checkpoint_path)  # Optionally save checkpoint path in trial attributes


        if scheduler:
            scheduler.step()
        # print(f'epoch: {epoch}, validation loss {validation_loss:.4f}')
        # Update the tqdm progress bar with the previous epoch's loss
        progress_bar.set_postfix(prev_loss=f'{previous_epoch_loss:.4e}')

        # Update the previous epoch's loss with the current validation loss
        previous_epoch_loss = validation_loss


        # Early stopping logic can be added here

    # Return the validation loss
    progress_bar.close()

    return validation_loss



study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
study.optimize(objective, n_trials=100)
