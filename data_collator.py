import transformers
from transformers import DataCollator
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

import numpy as np


class PCLDataCollator:
    def collate_batch(self, features):
        first = features[0]
        print(features)
        if isinstance(first, dict):
            if "label" in first and first["label"] is not None:
                if isinstance(first["label"], int):
                    labels = torch.tensor([f["label"]
                                           for f in features], dtype=torch.long)
                else:
                    labels = torch.tensor([f["label"]
                                           for f in features], dtype=torch.float)
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "label" and v is not None and not isinstance(v, str):
                    # print(k)
                    if isinstance(first[k], torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    else:
                        dtype = torch.long if type(
                            first[k]) is int else torch.float
                        batch[k] = torch.tensor(
                            [f[k] for f in features], dtype=dtype)
            return batch
        else:
            return transformers.DefaultDataCollator().collate_batch(features)

    def __call__(self, features):
        return self.collate_batch(features)


class StrIgnoreDevice(str):
    def to(self, _):
        return self


class DataLoaderWithTaskname:
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataLoader:
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]

        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)

        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataLoader({
            task_name: self.get_single_train_dataloader(
                task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })
