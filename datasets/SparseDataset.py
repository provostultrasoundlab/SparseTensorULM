import torch
from torch.utils.data import DataLoader
import numpy as np

from .GenericDataset import GenericDataset
import MinkowskiEngine as ME


def get_dataloader(train_dataset, val_dataset, train_params, val_params):
    train_loader = DataLoader(train_dataset, **train_params,
                              collate_fn=sparse2sparse_collation_fn)
    val_loader = DataLoader(val_dataset, **val_params,
                            collate_fn=sparse2sparse_collation_fn)
    return train_loader, val_loader


def sparse2sparse_collation_fn(data_labels):
    coords, feats, coords_labels, labels = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)
    bcoords_labels = ME.utils.batched_coordinates(coords_labels)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()

    return bcoords, feats_batch, bcoords_labels, labels_batch


class SparseDataset(GenericDataset):
    def __init__(self, *args, **kwargs):
        super(SparseDataset, self).__init__(*args, **kwargs)
        self.network_stride = 8

    def __getitem__(self, idx):
        group_key = str(idx + 1)  # Assuming groups are named 1, 2, 3, ...
        data_group = self.h5_file[group_key]
        corrMap = np.asarray(data_group['CorrMap'])
        corrMap = corrMap.T
        if self.thresholdSparsifier is not None:
            corrMap_abs = corrMap[0]**2 + corrMap[1]**2
            corrMap_abs = corrMap[0]**2 + corrMap[1]**2
            mask_sparse = (corrMap_abs > self.thresholdSparsifier)
            mask_sparse = torch.from_numpy(mask_sparse)
            mask_sparse = mask_sparse.unsqueeze(0).float().numpy()

        elif self.topk is not None:
            corrMap_abs = torch.from_numpy(corrMap[0]**2 + corrMap[1]**2)
            i_mask_sparse = torch.topk(corrMap_abs.reshape(-1),
                                       k=self.topk,
                                       dim=0)
            i_mask_sparse = i_mask_sparse.indices
            mask_sparse = torch.zeros(corrMap_abs.reshape(-1).shape)
            mask_sparse[i_mask_sparse] = 1
            mask_sparse = mask_sparse.reshape(1, *corrMap_abs.shape).numpy()
        elif self.mask_id is not None:
            mask_group = self.mask_h5_file[group_key]
            mask_sparse = np.asarray(mask_group[self.mask_id])
        else:
            raise NotImplementedError

        inputs = torch.from_numpy(corrMap) * mask_sparse

        inputs = inputs.squeeze()
        # permute dimension : first dimension goes to last, rest is unchanged.
        # pytorch dense dimension goes last
        inputs = torch.permute(inputs, list(range(1, len(inputs.shape)))+[0])
        # convert to sparse with 3 or 4 dimension and 1 features dim
        sparse_inputs = inputs.to_sparse(sparse_dim=len(inputs.shape)-1)
        # coalesce is required by pytorch
        sparse_inputs = sparse_inputs.coalesce()
        coords_x = sparse_inputs.indices()
        # need to swap axes for convention dicrepancy
        #  between torch and Minkowski Engine
        coords_x = torch.permute(coords_x, [1, 0]).numpy()
        # Dilate coordinate to account for stride in MinkowskiEngine
        coords_x[:, 0:-1] = self.network_stride * coords_x[:, 0:-1]
        feats_x = sparse_inputs.values().numpy()

        coords_y = np.asarray(data_group['ground_truth']['coords'])
        coords_y = coords_y.transpose(0, 2, 1)
        coords_y = coords_y.reshape(-1, 4)
        feats_y = np.asarray(np.ones((coords_y.shape[0], 1)))
        coords_y[:, :-1] = np.round(coords_y[:, :-1] * self.network_stride)
        coords_y[:, 3] = np.round(coords_y[:, 3] * 1e3)
        feats_y = np.asarray(np.ones((coords_y.shape[0], 1)))
        return coords_x, feats_x, coords_y, feats_y
