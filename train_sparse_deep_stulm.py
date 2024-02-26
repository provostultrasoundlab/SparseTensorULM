import json
import argparse
import os
import random
import numpy as np
from comet_ml import Experiment, OfflineExperiment
import time
from datasets.SparseDataset import SparseDataset, get_dataloader
from models.sparse_deepstULM import SparseULMunet
from trainers.SparseTrainer import SparseTrainer
from test_sparse_deep_stulm import test_in_silico
# torch need to be imported after
# Minkowskiengine (imported in dataset and trainer)
import torch


def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,
                        type=str, help='path to the config'
                        'file for the training')
    parser.add_argument('--save_path_test', default=None,
                        type=str, help='save path for the results')
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

    # ME introduces randomness anyway
    # (see https://github.com/NVIDIA/MinkowskiEngine/issues/504)
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
    # if no api key is detected the try statement raise
    # a value error, and we logg an offline experiment
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
    experiment.add_tag('sparse')
    # create the network
    network = SparseULMunet(**config['model'])

    # create the trainer, it keeps track of the training state
    trainer = SparseTrainer(network, **config['training'], config=config,
                            device=args.device,
                            experiment=experiment)

    # save the config
    trainer.save_config(config)
    # create datasets and dataloaders
    train_dataset = SparseDataset('train_set.h5', **config["data"])
    val_dataset = SparseDataset('val_set.h5', **config["data"])
    train_loader, val_loader = get_dataloader(train_dataset,
                                              val_dataset,
                                              **config['data_loaders'])
    n_epoch_done = 0
    for phase_idx, n_epoch_phase in enumerate(config['n_epochs']):

        for i_epoch in range(n_epoch_done, n_epoch_done + n_epoch_phase):
            start_time = time.time()
            if phase_idx == len(config['n_epochs'])-1:
                trainer.dilated = False
            if i_epoch > n_epoch_done:
                train_loader_ft, _ = get_dataloader(train_dataset,
                                                    val_dataset,
                                                    **config['data_loaders'])
                trainer.train(train_loader_ft)
                trainer.evaluate(val_loader)
            else:
                trainer.train(train_loader)
                # we don't validate in that case
            print('epoch duration', time.gmtime(time.time()-start_time))
        n_epoch_done += n_epoch_phase
        # refresh the lr (set to 0.1 initial_lr of phase 1)
        trainer.refresh_lr(config['refresh_factor'])
        trainer.on_phase_end()

    test_in_silico(config['training']['save_path'],
                   config['data_loaders']['val_params']["num_workers"],
                   config['data_loaders']['val_params']["batch_size"],
                   args.save_path_test)


if __name__ == "__main__":
    main()
