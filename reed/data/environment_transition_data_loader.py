#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import typing as t

import torch
from torch.utils.data import DataLoader

from reed.data.environment_transition_dataset import EnvironmentTransitionDataset, EnvironmentContrastiveBatch


class EnvironmentTransitionEnsembleDataLoader:
    """
    Handles loading and generating batches of preference triplets.

    The special logic needed is to handle different batch orderings for different networks in the reward ensemble
    """
    def __init__(self, dataset: EnvironmentTransitionDataset, ensemble_size: int,
                 batch_size: int = 64, num_workers: int = 0, shuffle: bool = True, device: torch.device = "cuda",
                 collate_fn: t.Optional = None):
        """
        Args:

        """
        # create a data loader per ensemble network
        self.loader_ensemble = [DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn)
                                for _ in range(ensemble_size)]

        self.device = device

    def __iter__(self) -> t.List[EnvironmentContrastiveBatch]:
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
                yield [next(dataloader_iterator) for dataloader_iterator in iter_loader_ensemble]
            except StopIteration:
                break
