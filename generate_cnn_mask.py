import os
import argparse
import json
import tqdm
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from datasets.DenseToSparseDataset import DenseToSparseDataset
from models.DenseToSparse import DenseToSparse


class SaverMaskDataset(DenseToSparseDataset):
    def __init__(self, *args, **kwargs):
        DenseToSparseDataset.__init__(self, *args, **kwargs)

    def __getitem__(self, index):
        corrMap, _, _ = DenseToSparseDataset.__getitem__(
            self, index)
        group_key = str(index + 1)
        return {"corrMap": corrMap,
                "group_key": group_key}


def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        type=str, help='path to the dense-to-sparse rundir')
    parser.add_argument('--mask_id', type=str, required=True,
                        help='string used to identify different versions'
                        ' of the cnn masks, used later in training config')
    parser.add_argument('--device', default='cuda',
                        type=str, help='device to use')

    args = parser.parse_args(raw_args)

    config = os.path.join(args.run_dir, 'config.json')
    with open(config) as json_file:
        config = json.load(json_file)

    # LOAD DENSE TO SPARSE NETWORK SWITCH TO EVAL, HALF AND CUDA
    network = DenseToSparse(**config['model'])
    model_path = os.path.join(args.run_dir, 'best_model')
    state_dict = torch.load(model_path)
    network.load_state_dict(state_dict['model_state_dict'])
    network = network.to(args.device)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda:{}'.format(torch.cuda.current_device())
    else:
        device = 'cpu'

    data_path = config['data']['data_path']
    dataset_names = ['train_set.h5', 'val_set.h5', 'test_set_1MB.h5',
                     'test_set_5MB.h5', 'test_set_10MB.h5', 'test_set_20MB.h5']

    for dataset_name in dataset_names:
        dataset = SaverMaskDataset(dataset_name, data_path)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=6)

        network.eval()

        mask_file = 'mask_'+dataset_name
        mask_path = os.path.join(data_path, mask_file)
        with h5py.File(mask_path, "a", libver='latest') as data:

            with torch.no_grad():
                for _, batch in tqdm.tqdm(enumerate(loader)):
                    corrMap = batch["corrMap"].float()
                    corrMap = corrMap.to(device)
                    mask = network(corrMap)
                    group_key = batch["group_key"][0]
                    if network.dim < 4:
                        mask = torch.nn.functional.interpolate(
                            mask.float(), size=corrMap.shape[2:])
                    if network.dim == 4:
                        assert mask.shape[0] == 1
                        assert mask.shape[2] == corrMap.shape[2]
                        mask = mask.float()
                        mask = interpolate(mask[0],
                                           size=corrMap.shape[3:])
                        mask = mask.unsqueeze(0)
                    mask = mask.cpu().numpy()
                    group = data.require_group(group_key)
                    # we try to create the group and to erase the mask
                    # in case it is already computed
                    try:
                        del data[group_key][args.mask_id]
                    except (ValueError, KeyError):
                        pass
                    group = data[group_key]
                    group.create_dataset(args.mask_id, dtype=np.int8,
                                         data=mask)


if __name__ == '__main__':
    main()
