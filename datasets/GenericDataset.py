from torch.utils.data import Dataset
import h5py
import os


class GenericDataset(Dataset):
    def __init__(self, filename, data_path, dilate=False,
                 thresholdSparsifier=None, topk=None,
                 mask_id=None):
        self.dilate = dilate
        self.h5_file = h5py.File(os.path.join(data_path, filename),
                                 'r', swmr=True)
        self.topk = topk
        self.thresholdSparsifier = thresholdSparsifier
        self.mask_id = mask_id
        if self.mask_id is not None:
            mask_file = os.path.join(data_path,
                                     'mask_'+filename)
            self.mask_h5_file = h5py.File(mask_file,
                                          'r', swmr=True)

    def __len__(self):
        return len(self.h5_file.keys())

    def __getitem__(self, idx):
        raise NotImplementedError

    def __del__(self):
        self.h5_file.close()
        if self.mask_id is not None:
            self.mask_h5_file.close()
