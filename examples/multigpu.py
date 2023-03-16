# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The DDP(Distributed Data Parallel) example using MS-AMP.

It is adapted from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import msamp


class MyTrainDataset(Dataset):
    """Training data set."""
    def __init__(self, size):
        """Constructor.

        Args:
            size: the first dimension of the data set.
        """
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        """Return the size of data set."""
        return self.size

    def __getitem__(self, index):
        """Retun data by index."""
        return self.data[index]


def ddp_setup(rank, world_size):
    """Setup DDP environment.

    Args:
        rank (int): Unique identifier of each process.
        world_size (int): Total number of processes.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


class Trainer:
    """Trainer class for DDP model."""
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        """Constructor.

        Args:
            model (torch.nn.Module): The model to train.
            train_data (DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer to update parameters.
            gpu_id (int): The GPU id.
            save_every (int): How often to save a checkpoint.
        """
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        """Run a batch of data.

        Args:
            source (torch.Tensor): The input data.
            targets (torch.Tensor): The target data.
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        """Run an epoch of training.

        Args:
            epoch (int): The epoch number.
        """
        b_sz = len(next(iter(self.train_data))[0])
        print(f'[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}')
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        """Save a checkpoint.

        Args:
            epoch (int): The epoch number.
        """
        ckp = self.model.module.state_dict()
        PATH = 'checkpoint.pt'
        torch.save(ckp, PATH)
        print(f'Epoch {epoch} | Training checkpoint saved at {PATH}')

    def train(self, max_epochs: int):
        """Train the model.

        Args:
            max_epochs (int): The maximum number of epochs to train.
        """
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    """Create and return training objects.

    Returns:
        train_set (Dataset): The dataset for training.
        model (torch.nn.Module): The model to train.
        optimizer (torch.nn.Optimizer): The optimizer for updating parameters.
    """
    train_set = MyTrainDataset(2048)    # load your dataset
    model = torch.nn.Linear(20, 1)    # load your model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model, optimizer = msamp.initialize(model, optimizer)

    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """Prepare data loader.

    Args:
        dataset (Dataset): The data set to use for training.
        batch_size (int): batch size.

    Returns:
        DataLoader: The data loader for training.
    """
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    """The entrypoint for training.

    Args:
        rank (int): The rank of the process.
        world_size (int): The number of processes in process group.
        save_every (int): How often to save a checkpoint.
        total_epochs (int): The total number of epochs to train.
        batch_size (int): The size of one batch.
    """
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
