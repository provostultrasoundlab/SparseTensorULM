import os
import tqdm
import numpy as np
from MinkowskiEngine import SparseTensor as SpTensor
import torch
import torch.nn as nn
from .trainer import GenericTrainer


class MultiscaleSparseDiceLoss(nn.Module):
    """
    Module use to compute the dice loss for several scale
    (corresponding to different stride) for dimension 3 or 4
    """

    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim

    def forward(self, inputs, targets, dilated=False, eval=False):
        smooth = 0.001
        res = []
        for level_idx in range(len(targets)):
            in_ = inputs[level_idx]
            target = targets[level_idx]

            # recreate sparse tensor to avoid index error in union map
            y_hat = SpTensor(coordinates=in_.coordinates,
                             features=in_.features[:, :1],
                             tensor_stride=in_.tensor_stride,
                             device=in_.device)

            y = SpTensor(coordinates=target.coordinates,
                         features=torch.ones((len(target.coordinates),
                                              1)),
                         tensor_stride=target.tensor_stride,
                         coordinate_manager=y_hat.coordinate_manager,
                         device=in_.device)

            # compute dilated ground truth for last level for first
            # stage of training in 2D similarly to deep-stULM
            if dilated and (level_idx == 3):
                # we will compute the dice in a dense formulation,
                #  using a sparse dice gives slighty different results
                y_shape = torch.Size([8, 1, 256, 256, 1])
                y_dense = y.dense(y_shape)[0]
                y_hat_dense = y_hat.dense(y_shape)[0]
                y_dense = torch.nn.functional.max_pool3d(
                    y_dense, kernel_size=[3, 3, 1],
                    stride=1, padding=[1, 1, 0])
                dice = 2*torch.sum(y_dense * y_hat_dense) / \
                    (torch.sum(y_dense) + torch.sum(y_hat_dense))
            else:
                # this is equivalent to elementwise multiplication as
                # it doesn't work properly in
                # ME see : https://github.com/NVIDIA/MinkowskiEngine/issues/390
                intersection = (torch.sum(torch.abs((y_hat+y).F)) -
                                torch.sum(torch.abs((y_hat-y).F)))/2
                dice = (2 * intersection + smooth) / \
                    (torch.sum(y.F) + torch.sum(y_hat.F)+smooth)
        # return as the dice loss : 1-dice
            res.append(1 - dice)
        return res


def compute_coords(coords, strides, dim):
    """
    Round the input coordinates based on the given strides
    and dimensions to obtain integer coordinates.

    This function takes a tensor of coordinates, a dimension
    value (either 3 or 4), and a tuple of strides.
    It adjusts the input coordinates by rounding them to
    the nearest multiple of the strides.
    If the dimension is 3, it stacks the first two
    and last two columns of the input coordinates (x, z axes).
    If the dimension is 4, the input coordinates
    remain unchanged (x, y, z axes).

    Args:
        coords (torch.Tensor): A tensor containing the input
        coordinates to be adjusted (x, y, z, t).
        dim (int): The dimension value, must be either 3
        or 4 (2D+t or 3D+t).
        strides (tuple): A tuple containing the strides
        to adjust the coordinates by (temporal_stride, spatial_stride).

    Returns:
        new_coords (torch.Tensor): A tensor containing
            the adjusted coordinates.

    Raises:
        ValueError: If the provided dimension value is not 3 or 4.
    """
    # Clone the input coordinates to avoid modifying the original tensor
    new_coords = coords.clone()
    # Adjust coordinates based on the dimension value
    if dim == 3:
        new_coords = torch.stack(
            [new_coords[:, 0], new_coords[:, 1],
             new_coords[:, 3], new_coords[:, 4]],
            dim=1)
    elif dim == 4:
        pass  # nothing to do here
    else:
        raise ValueError('incorrect dim')
    # Update the spatial dimensions of the coordinates by
    #  rounding them to the nearest multiple of the spatial stride
    new_coords[:, 1:-1] = strides[1] * \
        torch.floor(torch.div(new_coords[:, 1:-1], strides[1]))

    # Update the temporal dimension of the coordinates by rounding
    # them to the nearest multiple of the temporal stride
    new_coords[:, -1] = strides[0] * \
        torch.floor(torch.div(new_coords[:, -1], strides[0]))

    return new_coords


class SparseTrainer(GenericTrainer):
    def __init__(self, network, optimizer,
                 scheduler, config, save_path, device, experiment,
                 dilate_loss=False):
        super(SparseTrainer, self).__init__(network,
                                            optimizer,
                                            scheduler,
                                            config,
                                            save_path,
                                            device,
                                            experiment)
        self.criterion = MultiscaleSparseDiceLoss(dim=network.dim).to(device)
        self.dilated = dilate_loss

    def init_train(self):
        """
         method that initializes the training : switch to
         train mode, reset metrics, update strides
        """
        self.stride = self.network.stride
        temp_strides = self.network.temporal_strides
        spatial_strides = self.network.spatial_strides
        self.strides_list = list(zip(temp_strides, spatial_strides))
        self.stored_metrics = {}
        self.network.train()
        self.epoch += 1

    def train_step(self, batch):
        coords_x, feats_x, coords_y, feats_y = batch
        coords_x[:, 1:-1] = self.stride * torch.div(coords_x[:,  1:-1],
                                                    self.stride,
                                                    rounding_mode="floor")
        inputs = SpTensor(coordinates=coords_x,
                          features=feats_x,
                          tensor_stride=(
                              self.network.dim-1) * [self.stride] + [1],
                          device=self.device)
        self.optimizer.zero_grad()
        if self.network.deep_supervision:
            intermediate_gt = []
            for level_idx, strides in enumerate(self.strides_list):
                coords = compute_coords(coords_y, strides, self.network.dim)
                intermediate_gt.append({"coords_y": coords.clone(),
                                        "features_y": feats_y,
                                        "strides": strides
                                        })
            outputs, outs_pruning = self.network(inputs)
            outputs = [*outs_pruning, outputs]
            targets = []
            for level_idx, strides in enumerate(self.strides_list):
                if level_idx <= self.network.current_phase:
                    coords = compute_coords(
                        coords_y, strides, self.network.dim)
                    tensor_stride = (self.network.dim-1) * [strides[1]]
                    tensor_stride += [strides[0]]
                    coo_mngr = outputs[level_idx].coordinate_manager
                    target = SpTensor(coordinates=coords,
                                      features=feats_y,
                                      tensor_stride=tensor_stride,
                                      device=self.device,
                                      coordinate_manager=coo_mngr)
                    targets.append(target)
        else:
            outputs = [self.network(inputs)]
            strides = self.strides_list[-1]
            coords = compute_coords(coords_y, strides, self.network.dim)
            tensor_stride = (self.network.dim-1) * [strides[1]]
            tensor_stride += [strides[0]]
            coo_mngr = outputs[0].coordinate_manager
            targets = [SpTensor(coordinates=coords,
                                features=feats_y,
                                tensor_stride=tensor_stride,
                                device=self.device,
                                coordinate_manager=coo_mngr)]
        losses = self.criterion(outputs, targets, self.dilated)
        loss = torch.mean(torch.stack(losses))
        loss.backward()
        for loss_idx in range(min(self.network.current_phase+1, len(losses))):
            loss_name = "loss_{}".format(loss_idx)
            try:
                # append the loss to the stored metric
                if len(self.stored_metrics[loss_name]) < self.log_period:
                    self.stored_metrics[loss_name].append(
                        losses[loss_idx].item())
                # if enough loss values are stored average and log in comet
                else:
                    # log and empty the stored value
                    metric = self.stored_metrics[loss_name]
                    self.experiment.log_metric(loss_name,
                                               np.mean(metric),
                                               step=self.step)
                    self.stored_metrics[loss_name] = []
            # if the metric doesn't exist yet
            except KeyError:
                self.stored_metrics[loss_name] = []
        self.optimizer.step()
        return loss, losses

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
        dim = self.network.dim
        stride_input = self.network.stride
        strides = [self.network.temporal_strides[-1],
                   self.network.spatial_strides[-1]]
        loss = 0
        with self.experiment.test():
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader):
                    coords_x, feats_x, coords_y, feats_y = batch
                    coords_x[:, 1:-1] = self.stride * torch.div(coords_x[:,  1:-1],
                                                    self.stride,
                                                    rounding_mode="floor")
                    inputs = SpTensor(coordinates=coords_x,
                                    features=feats_x,
                                    tensor_stride=(
                                        self.network.dim-1) * [self.stride] + [1],
                                    device=self.device)
                    if self.network.deep_supervision:
                        output, _ = self.network(inputs)
                    else:
                        output = self.network(inputs)

                    for level_idx, strides in enumerate(self.strides_list):
                        if level_idx <= self.network.current_phase:
                            coords = compute_coords(
                                coords_y, strides, self.network.dim)
                            tar_stride = (dim-1) * [strides[1]]
                            tar_stride += [strides[0]]
                            coo_mngr = output.coordinate_manager
                            target = SpTensor(coordinates=coords,
                                              features=feats_y,
                                              tensor_stride=tar_stride,
                                              device=self.device,
                                              coordinate_manager=coo_mngr)

                    loss += self.criterion([output], [target])[0].item()
                    torch.cuda.empty_cache()
        loss /= len(val_loader)
        if loss < self.best_eval_loss:
            torch.save({'model_state_dict': self.network.state_dict()},
                       os.path.join(self.save_path, 'best_model'))
        print('validation loss : {:%}'.format(loss))
