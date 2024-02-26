
import os
import json
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


class GenericTrainer():
    """
    Abstract class used for DenseToSparse cnn and sparse model training
    """

    def __init__(self, network, optimizer,
                 scheduler, config, save_path, device, experiment):

        self.device = device
        self.network = network.to(device)
        self.optimizer_config = optimizer
        self.optimizer = Adam(network.parameters(),
                              **optimizer)
        self.scheduler_config = scheduler
        self.scheduler = MultiStepLR(self.optimizer,
                                     **scheduler)
        self.config = config
        self.epoch = 0
        self.step = 0
        self.experiment = experiment
        self.best_eval_loss = 1
        self.save_path = save_path

    def train(self, train_loader, log_period=10):
        """
        Argument : training loader, network, optimizer,
        criterion, LR scheduler, parameters dictionnary, graphs matrix to save
        losses.
        Output : none

        Training process for one epoch : for every batch in the data loader,
        infer, compute loss, backward optimization, some display
        """
        self.log_period = log_period
        train_loss = 0
        self.init_train()
        print('EPOCH {} '.format(self.epoch))
        with self.experiment.train():
            for batch_idx, batch in enumerate(train_loader):
                loss, losses = self.train_step(batch)
                train_loss += loss.item()
                message_args = []
                for i_loss, loss in enumerate(losses):
                    message_args.append(i_loss)
                    message_args.append(loss.item())
                if batch_idx % log_period == 0:
                    msg = ('batch {}/{}'+len(losses)*" | loss level {} : {:%}")
                    print(msg.format(batch_idx,
                                     len(train_loader),
                                     *message_args))

                torch.cuda.empty_cache()
                self.step += 1
            torch.save({'model_state_dict': self.network.state_dict()},
                       os.path.join(self.save_path, 'current_model'))
            torch.save({'optimizer_state_dict': self.optimizer.state_dict()},
                       os.path.join(self.save_path, 'current_optimizer'))
            torch.save({'scheduler_state_dict': self.scheduler.state_dict()},
                       os.path.join(self.save_path, 'current_scheduler'))
            train_loss /= len(train_loader)
            self.experiment.log_metric("learning_rate",
                                       self.scheduler.get_last_lr()[0],
                                       epoch=self.epoch)
            self.scheduler.step()

    def evaluate(self, val_loader):
        pass

    def init_train(self):
        pass

    def train_step(self, batch):
        pass

    def save_config(self, config):
        with open(os.path.join(self.save_path, 'config.json'), 'w') as file:
            json.dump(config, file, indent=4)

    def on_phase_end(self):
        self.best_eval_loss = 1
        self.network.current_phase += 1

    def refresh_lr(self, factor=0.1):
        self.optimizer_config['lr'] *= factor
        self.optimizer = Adam(self.network.parameters(),
                              **self.optimizer_config)
        self.scheduler = MultiStepLR(self.optimizer,
                                     **self.scheduler_config)
