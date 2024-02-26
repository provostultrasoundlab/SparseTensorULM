import numpy as np
import torch
import torch.nn.functional as F
from .GenericDataset import GenericDataset


class DenseDataset(GenericDataset):
    def __getitem__(self, idx):
        """ we inherit every method except for get item """
        group_key = str(idx + 1)  # Assuming groups are named 1, 2, 3, ...
        data_group = self.h5_file[group_key]
        corrMap = np.asarray(data_group['CorrMap'])
        corrMap = corrMap.T
        if self.thresholdSparsifier is not None:
            corrMap_abs = corrMap[0]**2 + corrMap[1]**2
            mask_gt = (corrMap_abs > self.thresholdSparsifierF)
            mask_gt = torch.from_numpy(mask_gt)
            mask_gt = mask_gt.unsqueeze(0).float().numpy()

        elif self.topk is not None:
            corrMap_abs = torch.from_numpy(corrMap[0]**2 + corrMap[1]**2)
            i_mask_gt = torch.topk(corrMap_abs.reshape(-1),
                                   k=self.topk,
                                   dim=0)
            i_mask_gt = i_mask_gt.indices
            mask_gt = torch.zeros(corrMap_abs.reshape(-1).shape)
            mask_gt[i_mask_gt] = 1
            mask_gt = mask_gt.reshape(1, *corrMap_abs.shape).numpy()
        else:
            mask_gt = np.ones_like(corrMap)
        corrMap = corrMap*mask_gt
        coords_y = np.asarray(data_group['ground_truth']['coords'])
        mask = np.zeros((8 * corrMap.shape[-3],
                         8 * corrMap.shape[-2]))
        coords_y = np.round(coords_y * 8).astype('int')
        mask[coords_y[:, 0], coords_y[:, 2]] = 1
        mask = mask.reshape(1, 1, *mask.shape)
        mask = torch.from_numpy(mask)
        if self.dilate:
            mask = F.max_pool2d(mask,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        corrMap = torch.from_numpy(corrMap)
        corrMap = corrMap.permute(0, 3, 1, 2).float()
        mask = mask.float()
        return corrMap, mask
