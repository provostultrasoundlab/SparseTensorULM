import numpy as np
import MinkowskiEngine as ME
import torch
import torch.utils.data as data

from .GenericDataset import GenericDataset


def dense2sparse_collation_fn(data_labels):
    try:
        corrMaps, coords_labels, labels = list(zip(*data_labels))
        batchedCoorMaps = torch.from_numpy(np.asarray(corrMaps)).float()
        # Create batched coordinates for the SparseTensor input
        bcoords_labels = ME.utils.batched_coordinates(coords_labels)
        # Concatenate all lists
        labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()
        return batchedCoorMaps, bcoords_labels, labels_batch
    except ValueError:
        corrMaps = np.stack(data_labels)
        batchedCoorMaps = torch.from_numpy(np.asarray(corrMaps)).float()
        return batchedCoorMaps


def get_dataloader(train_dataset, val_dataset, train_params, val_params):
    train_loader = data.DataLoader(
        train_dataset, **train_params, collate_fn=dense2sparse_collation_fn)
    val_loader = data.DataLoader(
        val_dataset, **val_params, collate_fn=dense2sparse_collation_fn)
    return train_loader, val_loader


class DenseToSparseDataset(GenericDataset):
    def __getitem__(self, idx):
        group_key = str(idx + 1)  # Assuming groups are named 1, 2, 3, ...
        data_group = self.h5_file[group_key]
        corrMap = np.asarray(data_group['CorrMap'])
        corrMap = corrMap.T
        try:
            coords_y = np.asarray(data_group['ground_truth']['coords'])
            coords_y = coords_y.transpose(0, 2, 1)
            coords_y = coords_y.reshape(-1, 4)
            feats_y = np.asarray(np.ones((coords_y.shape[0], 1)))
            # conversion from s to frame number
            coords_y[:, 3] = np.round(coords_y[:, 3] * 1e3)
            # ie multiplication by the frame rate of the simu
            return corrMap, coords_y, feats_y
        except KeyError:
            return corrMap
