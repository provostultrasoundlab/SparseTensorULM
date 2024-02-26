import os
import MinkowskiEngine as ME
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from trainers.trainer import GenericTrainer


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.
        intersection = input * target
        # if we compute the global dice then we will sum over the batch dim,
        # otherwise no
        dim = list(range(0, len(input.shape)))
        intersection = intersection.sum(dim=dim)
        card_pred = input.sum(dim)
        card_target = target.sum(dim)

        # return 1-dice for the loss
        res = 1 - ((2. * intersection + smooth) /
                   (card_pred + card_target + smooth))
        return res.mean()


def compute_coords(coords, spatial_stride, temp_stride, corrMap_shape, dim):
    if dim == 3:
        coords = torch.stack(
            [coords[:, 0], coords[:, 1], coords[:, 3], coords[:, 4]], dim=1)
    elif dim == 4:
        pass
    else:
        raise ValueError('incorrect dim')
    coords[:, 1:-1] = torch.div(coords[:, 1:-1],
                                spatial_stride,
                                rounding_mode='floor')
    coords[:, 1:-1] = spatial_stride * coords[:, 1:-1]
    coords[:, 1:-1] = torch.clip(coords[:, 1:-1],
                                 0,
                                 spatial_stride * corrMap_shape[-2] - 1)
    coords[:, -1] = torch.div(coords[:, -1],
                              temp_stride,
                              rounding_mode='floor')

    coords[:, -1] = torch.clip(temp_stride * coords[:, -1], 0,
                               corrMap_shape[-1] - 1)
    return coords


class DenseToSparseTrainer(GenericTrainer):
    def __init__(self, network, optimizer,
                 scheduler, config, save_path, device, experiment):
        super(DenseToSparseTrainer, self).__init__(network,
                                                   optimizer,
                                                   scheduler,
                                                   config,
                                                   save_path,
                                                   device,
                                                   experiment)
        self.criterion = DiceLoss().to(device)
        self.spatial_stride = 1
        # Creates a GradScaler once at the beginning of training.
        self.scaler = GradScaler()

    def init_train(self):
        """
         method that initializes the training : switch to train mode,
           reset metrics, update strides
        """
        self.network.train()
        self.epoch += 1

    def train_step(self, batch):
        corrMap, coords_y, feats_y = batch
        corrMap = corrMap.to(self.device)
        self.optimizer.zero_grad()
        with autocast(dtype=torch.float16):
            mask = self.network(corrMap)
        coords = compute_coords(coords_y.clone(), self.spatial_stride,
                                self.network.temporal_strides,
                                corrMap.shape, self.network.dim)
        # add spatial stride
        stride = (self.network.dim-1) * [self.spatial_stride]
        # add temporal stride
        stride += [self.network.temporal_strides]
        # compute the mask as a ME sparse tensor
        mask_gt = ME.SparseTensor(coordinates=coords,
                                  features=feats_y,
                                  tensor_stride=stride,
                                  device=self.device)

        mask_gt = mask_gt.dense(mask.shape)[0]
        with autocast(dtype=torch.float16):
            loss = self.criterion(mask, mask_gt)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        # Updates the scale for next iteration.
        self.scaler.update()

        return loss, [loss]

    def evaluate(self, val_loader):
        """
        Argument : training loader, current epoch number, network, optimizer,
        criterion, LR scheduler, parameters dictionnary, graphs matrix
        to save losses
        Output : none

        Evaluation process for one epoch : for every batch in the validation
        loader, infer, compute loss, save it, some display
        """
        self.network.eval()
        loss = 0
        with self.experiment.test():
            with torch.no_grad():
                for batch in val_loader:
                    corrMap, coords_y, feats_y = batch
                    corrMap = corrMap.to(self.device)
                    mask = self.network(corrMap)
                    coords = compute_coords(coords_y.clone(),
                                            self.spatial_stride,
                                            self.network.temporal_strides,
                                            corrMap.shape, self.network.dim)
                    # add spatial stride
                    stride = (self.network.dim-1) * [self.spatial_stride]
                    # add temporal stride
                    stride += [self.network.temporal_strides]
                    # compute the mask as a ME sparse tensor
                    mask_gt = ME.SparseTensor(coordinates=coords,
                                              features=feats_y,
                                              tensor_stride=stride,
                                              device=self.device)

                    mask_gt = mask_gt.dense(mask.shape)[0]
                    loss += self.criterion(mask, mask_gt).item()

                    torch.cuda.empty_cache()
        loss /= len(val_loader)
        if loss < self.best_eval_loss:
            torch.save({'model_state_dict': self.network.state_dict()},
                       os.path.join(self.save_path, 'best_model'))
        print('validation loss : {:%}'.format(loss))
