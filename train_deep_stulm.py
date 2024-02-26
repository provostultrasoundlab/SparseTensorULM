import os
import time
import json
import numpy as np
from comet_ml import Experiment, OfflineExperiment
import torch
import torch.nn as nn
import torch.utils.data as data
import argparse
import random
from models.deepSTULM import ulmVNetMax
from datasets.DenseDataset import DenseDataset
from test_deep_stulm import test_in_silico


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))


def train(train_loader, epoch, network, optimizer, criterion,
          device, experiment):
    network.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            msg = msg.format(epoch, batch_idx * len(inputs),
                             len(train_loader.dataset),
                             100. * batch_idx / len(train_loader))
            msg += '| Batch Loss: {:.3e}'.format(loss.item())
            print(msg)

    train_loss /= len(train_loader)
    print('\nTrain set - Average loss: {:.4e}'.format(train_loss))
    experiment.log_metric("training loss", train_loss,
                          epoch=epoch, include_context=False)


def evaluate(eval_loader, epoch, network, criterion, device, experiment):
    network.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            eval_loss += criterion(outputs, targets).item()
            outputs = torch.sigmoid(outputs)

    eval_loss /= len(eval_loader)
    print('Validation set - Average loss: {:.4e}'.format(eval_loss))
    experiment.log_metric("validation loss", eval_loss, include_context=False)
    return eval_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,
                        type=str, help='path to the config'
                        'file for the training')
    parser.add_argument('--data_prefix', required=True,
                        type=str, help='path to data dir')
    parser.add_argument('--seed', default=0,
                        type=int, help='seed for reproductibility')
    parser.add_argument('--save_path_test', default=None,
                        type=str, help='save path for the results')
    parser.add_argument('--tags', default=None,
                        type=str, help='tag to add to cometml')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # fix seed for reproductibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = args.config_path
    data_prefix = args.data_prefix

    # check and default the config file, keep it compliant
    # with the change in the config file
    try:
        experiment = Experiment(project_name='sparsetensorulm',
                                workspace='bricerauby', auto_log_co2=False)
    # if no api key is detected the try statement raise a value error,
    # and we log an offline experiment
    except ValueError:
        offline_directory = "/home/raubyb/scratch/cometml"
        os.makedirs(offline_directory, exist_ok=True)
        experiment = OfflineExperiment(project_name='sparsetensorulm',
                                       workspace='bricerauby',
                                       offline_directory=offline_directory)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        experiment.set_name(timestr)
    experiment.set_name(os.environ.get('SLURM_JOB_ID') +
                        '_' + experiment.get_name())

    if args.tags is not None:
        experiment.add_tag(args.tags)
    experiment.add_tag('Deep-stULM')

    with open(config_path) as json_file:
        config = json.load(json_file)

    config['data']['data_path'] = os.path.join(data_prefix,
                                               config['data']['data_path'])
    path_save = os.path.join(config['saver']['path'], experiment.name)
    os.makedirs(path_save, exist_ok=True)

    with open(os.path.join(path_save, 'config.json'), 'w') as file:
        json.dump(config, file, indent=4)

    data_path = config['data']['data_path']
    train_dataset = DenseDataset('train_set.h5', data_path)
    train_dataset_dilated = DenseDataset(
        'train_set.h5', data_path, dilate=True)
    val_dataset = DenseDataset('val_set.h5', data_path)
    train_loader = data.DataLoader(
        train_dataset, **config['data_loaders']['train_params'],
        shuffle=True, pin_memory=True)
    train_loader_dilated = data.DataLoader(
        train_dataset_dilated, **config['data_loaders']['train_params'],
        shuffle=True, pin_memory=True)

    val_loader = data.DataLoader(
        val_dataset, **config['data_loaders']['val_params'],
        shuffle=False, pin_memory=True)

    dropout = False
    network = ulmVNetMax(elu=True, dropout=dropout).to(device)

    print(network)
    total_params = sum(p.numel() for p in network.parameters())
    print('/nNumber of parameters of the network : {}'.format(total_params))

    # PHASE 1
    LR = 1e-1
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = DiceLoss()
    gamma = 0.1
    milestones = [15, 45, 75, 100]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=gamma)

    best_eval_loss = 1
    for epoch in range(1, config["n_epochs"][0]+1):
        print('EPOCH {}'.format(epoch) +
              ' - LR = {:.0E}'.format(scheduler.get_last_lr()[0]))

        train(train_loader_dilated, epoch, network,
              optimizer, criterion, device, experiment)
        eval_loss = evaluate(val_loader, epoch, network,
                             criterion, device, experiment)

        torch.save({'model_state_dict': network.state_dict()},
                   os.path.join(path_save, 'current_model'))
        torch.save({'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(path_save, 'current_optimizer'))
        torch.save({'scheduler_state_dict': scheduler.state_dict()},
                   os.path.join(path_save, 'current_scheduler'))

        if eval_loss < best_eval_loss:
            torch.save({'model_state_dict': network.state_dict()},
                       os.path.join(path_save, 'best_model'))
        if scheduler is not None:
            scheduler.step()

    LR = 1e-3
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = DiceLoss()
    gamma = 0.1
    milestones = [10, 50, 100]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=gamma)

    for epoch in range(config["n_epochs"][0]+1,
                       config["n_epochs"][0] + config["n_epochs"][1] + 1):
        print('EPOCH {}'.format(epoch) +
              ' - LR = {:.0E}'.format(scheduler.get_last_lr()[0]))

        train(train_loader, epoch, network, optimizer,
              criterion, device, experiment)

        eval_loss = evaluate(val_loader, epoch, network,
                             criterion, device, experiment)

        torch.save({'model_state_dict': network.state_dict()},
                   os.path.join(path_save, 'current_model'))
        torch.save({'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(path_save, 'current_optimizer'))
        torch.save({'scheduler_state_dict': scheduler.state_dict()},
                   os.path.join(path_save, 'current_scheduler'))

        if eval_loss < best_eval_loss:
            torch.save({'model_state_dict': network.state_dict()},
                       os.path.join(path_save, 'best_model'))

        if scheduler is not None:
            scheduler.step()
    # Testing in silico
    test_in_silico(path_save,
                   config['data_loaders']['val_params']["num_workers"],
                   config['data_loaders']['val_params']["batch_size"],
                   args.save_path_test)
