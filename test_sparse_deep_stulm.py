import os
import argparse
import torch
import torch.utils.data as data
import numpy as np
import json
import tqdm
import MinkowskiEngine as ME
from datasets.SparseDataset import SparseDataset, sparse2sparse_collation_fn
from models.sparse_deepstULM import SparseULMunet
from trainers.SparseTrainer import compute_coords
import hdf5storage


def evaluate(concentration, config, network, output_dir,
             batch_size, num_workers, device):
    filename = 'test_set_{}.h5'.format(
        concentration)
    dataset = SparseDataset(filename, **config["data"])
    loader = data.DataLoader(dataset, num_workers=num_workers,
                             batch_size=batch_size,
                             shuffle=False, pin_memory=True,
                             drop_last=True,
                             collate_fn=sparse2sparse_collation_fn)
    output_dir = os.path.join(
        output_dir, '{}_{}'.format(concentration, 'test'))
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    network = network.eval()
    dim = network.dim
    output_size = [loader.batch_size, 1]
    output_size += (dim-1)*[256] + [1]
    output_size = torch.Size(output_size)

    temp_strides = network.temporal_strides
    spatial_strides = network.spatial_strides
    strides_list = list(zip(temp_strides, spatial_strides))
    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
            coords_x, feats_x, coords_y, feats_y = batch
            coords_x[:, 1:-1] = torch.div(coords_x[:, 1:-1],
                                          network.spatial_strides[0],
                                          rounding_mode="floor")
            coords_x[:, 1:-1] = network.spatial_strides[0] * coords_x[:, 1:-1]
            tensor_stride = (dim-1) * [network.spatial_strides[0]] + [1]
            inputs = ME.SparseTensor(coordinates=coords_x,
                                     features=feats_x,
                                     tensor_stride=tensor_stride,
                                     device=device)
            try:
                outputs, _ = network(inputs)
            except TypeError:
                outputs = network(inputs)
            strides = strides_list[-1]
            coords = compute_coords(coords_y, strides, network.dim)
            coordinate_manager = outputs.coordinate_manager
            targets = ME.SparseTensor(coordinates=coords,
                                      features=feats_y[:, :1],
                                      tensor_stride=(
                                          dim-1) * [strides[1]] + [strides[0]],
                                      device=device,
                                      coordinate_manager=coordinate_manager)
            pred = (outputs.dense(output_size)[0] > 0.5).int()
            pred = pred.cpu().numpy().squeeze()
            gt = (targets.dense(output_size)[0] > 0.5).int()
            gt = gt.cpu().numpy().squeeze()
            torch.cuda.empty_cache()
            save_path = os.path.join(output_dir,
                                     'pred_{}.mat'.format(batch_idx))
            hdf5storage.write({"prediction_SparseNeST": pred.squeeze(),
                               "ground_truth": gt.squeeze()},
                              filename=save_path, matlab_compatible=True)
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


def test_in_silico(run_dir, num_workers, batch_size, save_path):

    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path) as json_file:
        config = json.load(json_file)
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(torch.cuda.current_device())
    else:
        device = 'cpu'
    # Load the network
    network = network = SparseULMunet(**config['model'])
    model_path = os.path.join(run_dir, 'best_model')
    state_dict = torch.load(model_path)
    network.current_phase = 4
    network.load_state_dict(state_dict["model_state_dict"])
    network.eval()
    network.to(device)

    output_dir = os.path.join(save_path, run_dir)
    os.makedirs(output_dir, exist_ok=True)
    if network.dim == 3:
        evaluate('1MB', config, network, output_dir,
                 batch_size, num_workers, device)
        evaluate('5MB', config, network, output_dir,
                 batch_size, num_workers, device)
        evaluate('10MB', config, network, output_dir,
                 batch_size, num_workers, device)
        evaluate('20MB', config, network, output_dir,
                 batch_size, num_workers, device)
    if network.dim == 4:
        evaluate('10MB', config, network, output_dir,
                 batch_size, num_workers, device)
        evaluate('30MB', config, network, output_dir,
                 batch_size, num_workers, device)
        evaluate('1MB', config, network, output_dir,
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


# python test_sparse_deep_stulm.py --run_dir runs_3D_sparse_sparse_only_20_12/20818091_zealous_cuisine_4538 --num_workers 4 --batch_size 4 --save_path 