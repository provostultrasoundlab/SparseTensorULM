import json
import argparse
import os
import numpy as np
from time import time
import random

from comet_ml import Experiment, OfflineExperiment
import torch

from models.DenseToSparse import DenseToSparse
from datasets.DenseToSparseDataset import DenseToSparseDataset, get_dataloader
from trainers.DenseToSparseTrainer import DenseToSparseTrainer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,
                        type=str, help='path to the config'
                        'file for the training')
    parser.add_argument('--data_prefix', required=True,
                        type=str, help='path to data dir')
    parser.add_argument('--device', default='cuda:0',
                        type=str, help='device to use')
    parser.add_argument('--seed', default=0,
                        type=int, help='seed to use')
    parser.add_argument('--tags', default=None,
                        type=str, help='tag to add to cometml')
    args = parser.parse_args(raw_args)

    with open(args.config_path) as json_file:
        config = json.load(json_file)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print('random seed is ', args.seed)

    config['data']['data_path'] = os.path.join(args.data_prefix,
                                               config['data']['data_path'])

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
    config['training']['save_path'] = os.path.join(
        config['training']['save_path'], experiment.name)
    os.makedirs(config['training']['save_path'], exist_ok=True)
    if args.tags is not None:
        experiment.add_tag(args.tags)
    experiment.add_tag('dense-to-sparse-cnn')

    # create the network
    network = DenseToSparse(**config['model'])

    # create the trainer, it keeps track of the training state
    trainer = DenseToSparseTrainer(network,
                                   **config['training'],
                                   config=config,
                                   device=args.device,
                                   experiment=experiment)

    # save the config
    trainer.save_config(config)

    # create datasets and dataloaders
    data_path = config['data']['data_path']
    train_dataset = DenseToSparseDataset('train_set.h5', data_path)
    val_dataset = DenseToSparseDataset('val_set.h5', data_path)
    train_loader, val_loader = get_dataloader(train_dataset,
                                              val_dataset,
                                              **config['data_loaders'])
    n_epoch_done = 0
    for _, n_epoch in enumerate(config['n_epochs']):
        for _ in range(n_epoch_done, n_epoch_done + n_epoch):
            trainer.train(train_loader)
            trainer.evaluate(val_loader)
        n_epoch_done += n_epoch


if __name__ == "__main__":
    main()
