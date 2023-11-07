#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import typing as t
import attr

import torch
from torch.utils.data import DataLoader

from reed.data.preference_dataset import PreferenceDataset


@attr.s
class PreferenceTripletBatch:
    trajectories_one = attr.ib(type=torch.Tensor)
    trajectories_two = attr.ib(type=torch.Tensor)
    preference_labels = attr.ib(type=torch.Tensor)


UNFORMATTED_PREFERENCE_TRIPLET_BATCH = t.List[t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
FORMATTED_PREFERENCE_TRIPLET_BATCH = t.List[PreferenceTripletBatch]


class PreferenceTripletEnsembleDataLoader:
    """
    Handles loading and generating batches of preference triplets.

    The special logic needed is to handle different batch orderings for different networks in the reward ensemble
    """
    def __init__(self, dataset: PreferenceDataset, ensemble_size: int,
                 batch_size: int = 64, num_workers: int = 0, shuffle: bool = True, device: torch.device = "cuda"):
        """
        Args:

        """
        # create a data loader per ensemble network
        self.loader_ensemble = [DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers)
                                for _ in range(ensemble_size)]

        self.device = device

    def _format_batch(self, batch: UNFORMATTED_PREFERENCE_TRIPLET_BATCH) -> FORMATTED_PREFERENCE_TRIPLET_BATCH:
        """
        Format the preference batch so that the tensors are longs and on the correct device
        """
        return [PreferenceTripletBatch(trajectories_one=member[0].float().to(self.device),
                                       trajectories_two=member[1].float().to(self.device),
                                       preference_labels=member[2].long().to(self.device))
                for member in batch]

    def dataset_length(self) -> int:
        return len(self.loader_ensemble[0].dataset)

    def __iter__(self) -> FORMATTED_PREFERENCE_TRIPLET_BATCH:
        """
        Iterate through the preference triplet data loaders and return the batch per ensemble member

        Returns:
            list of PreferenceTripletBatch
        """
        # set up each loader as an iterator
        iter_loader_ensemble = [iter(loader) for loader in self.loader_ensemble]
        # for each data loader grab the next batch until there are no more batches to grab
        while True:
            # check if there is a next batch to return
            try:
                yield self._format_batch([next(dataloader_iterator) for dataloader_iterator in iter_loader_ensemble])
            except StopIteration:
                break
