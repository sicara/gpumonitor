"""
Script taken from ["From PyTorch to PyTorch Lightning - A gentle introduction"](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
"""
import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

from gpumonitor.callbacks import PyTorchGpuMonitorCallback


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()

        self.layer_1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = torch.log_softmax(self.layer_1(x), dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = LightningMNISTClassifier()
trainer = pl.Trainer(
    max_epochs=1,
    gpus=1,
    callbacks=[PyTorchGpuMonitorCallback(delay=0.5)]
)

trainer.fit(model)
