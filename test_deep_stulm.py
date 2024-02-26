import os
import argparse
import torch
import torch.utils.data as data
import numpy as np
import json
import tqdm
from models.deepSTULM import ulmVNetMax
from datasets.DenseDataset import DenseDataset
import hdf5storage


def evaluate(concentration, data_path, network,
             output_dir, batch_size,
             num_workers, device):
    dataset = DenseDataset('test_set_{}.h5'.format(concentration), data_path)
    loader = data.DataLoader(dataset, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False,
                             pin_memory=True, drop_last=True)
    output_dir = os.path.join(
        output_dir, '{}_{}'.format(concentration, 'test'))
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    network = network.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            pred = (outputs > 0).int().cpu().numpy().squeeze()
            gt = (targets > 0.5).int().cpu().numpy().squeeze()
            torch.cuda.empty_cache()
            save_path = os.path.join(output_dir,
                                     'pred_{}.mat'.format(batch_idx))
            hdf5storage.write({"prediction_DeepST": pred.squeeze(),
                               "ground_truth": gt.squeeze()},
                              filename=save_path,
                              matlab_compatible=True)
            if batch_idx == 0:
                acc_pred = np.sum(pred, axis=0)
                acc_gt = np.sum(gt, axis=0)
            else:
                acc_pred += np.sum(pred, axis=0)
                acc_gt += np.sum(gt, axis=0)
        # compute dice
        acc_gt = (acc_gt > 0.5).astype('int')
        acc_pred = (acc_pred > 0.5).astype('int')
        intersection = np.sum(acc_gt * acc_pred)
        card_gt = np.sum(acc_gt)
        card_pred = np.sum(acc_pred)
        dice = 2 * intersection/(card_gt + card_pred + 1e-8)
        print('dice on test set is :', dice)
        print('dice on test set is :', dice)


def test_in_silico(run_dir, num_workers, batch_size, save_path):

    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path) as json_file:
        config = json.load(json_file)
    data_path = config["data"]["data_path"]
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(torch.cuda.current_device())
    else:
        device = 'cpu'
    # Load the network
    network = ulmVNetMax(elu=True, dropout=False).to(device)
    model_path = os.path.join(run_dir, 'best_model')
    state_dict = torch.load(model_path)
    network.load_state_dict(state_dict["model_state_dict"])
    network.eval()
    network.to(device)

    output_dir = os.path.join(save_path, run_dir)
    os.makedirs(output_dir, exist_ok=True)
    evaluate('1MB', data_path, network, output_dir,
             batch_size, num_workers, device)
    evaluate('5MB', data_path, network, output_dir,
             batch_size, num_workers, device)
    evaluate('10MB', data_path, network, output_dir,
             batch_size, num_workers, device)
    evaluate('20MB', data_path, network, output_dir,
             batch_size, num_workers, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        type=str, help='path to the directory of the run')
    parser.add_argument('--num_workers', required=True,
                        type=int, help='number of workers for the dataloader')
    parser.add_argument('--batch_size', required=True,
                        type=int, help='batch size for the inference')
    parser.add_argument('--save_path', required=True,
                        type=str, help='path to save directory')
    args = parser.parse_args()

    test_in_silico(args.run_dir, args.num_workers,
                   args.batch_size, args.save_path)
