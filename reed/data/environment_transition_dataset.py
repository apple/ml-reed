#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import typing as t
import attr
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import PackedSequence

from BPref.replay_buffer import TrajectoryReplayBuffer


@attr.s
class EnvironmentContrastiveDatapoint:
    """
    A triplet where two states/observations are given and one is an augmented version of the other.

    The augmentation may be along the lines of random crop, jitter, etc or may be a temporal augmentation where the
    augmented state occurs in the future
    """
    state = attr.ib(type=torch.Tensor)
    action = attr.ib(type=torch.Tensor)
    augmented_state = attr.ib(type=torch.Tensor)


@attr.s
class EnvironmentContrastiveBatch:
    """
    A batch of triplets where two states/observations are given and one is an augmented version of the other.

    The augmentation may be along the lines of random crop, jitter, etc or may be a temporal augmentation where the
    augmented state occurs in the future
    """
    states = attr.ib(type=t.Union[torch.Tensor, PackedSequence])
    actions = attr.ib(type=t.Union[torch.Tensor, PackedSequence])
    augmented_states = attr.ib(type=t.Union[torch.Tensor, PackedSequence])

    def to_dict(self) -> t.Mapping[str, t.Union[torch.Tensor, PackedSequence]]:
        """
        Return the attr as a dictionary
        """
        return {"states": self.states,
                "actions": self.actions,
                "augmented_states": self.augmented_states}


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class EnvironmentTransitionDataset(Dataset):
    """
    A dataset of environment transitions where the state-action pairs are inputs
    and the next states are the target values.

    The dataset can be loaded from a file saved to disk or from a Replay Buffer.

    States and next states can be images.
    """
    def __init__(self, replay_buffer: t.Optional[TrajectoryReplayBuffer] = None,
                 file_path: t.Optional[Path] = None,
                 target: str = "next_observation",
                 device: str = "cuda",
                 multi_gpu: bool = False,
                 image_observations: bool = False,
                 image_formatter: t.Optional[t.Any] = None):
        """
        Either the replay_buffer or the file_path needs to not be of type None. If neither are of type
        None then both are used to populate the dataset

        Args:
            replay_buffer: the buffer of collected environment transitions
            file_path: the location of the datasets file
            target: (default = next_observation) the target for the SFC objective. Must be one of next_observation
                    (the target is the next observation) or observation_difference (the target is the difference between
                    the current and next observation).
            device: (default = cuda) whether to run on the cpu or a cuda device
            multi_gpu: (default = False) whether the model is trained across multiple GPUs in which case we do not
                       push the data to a device before returning it
            image_observations: (default = False) whether or not the states are to be tracked as images
            image_formatter: (default = None) a function to apply to the raw images in order to format them them for
                             training
                TODO: define expected format for this file will probably do this
                      once get to the point of writing out a dataset file.
        """
        assert replay_buffer is not None or file_path is not None, ("One of replay_buffer or file_path must be "
                                                                    "specified. Both are None.")
        super(EnvironmentTransitionDataset, self).__init__()

        assert target in {"next_observation", "observation_difference"}, (f"target must be one of 'next_observation' or"
                                                                          f" 'observation_difference', not {target}.")
        self._target = target

        self.states: t.Optional[np.ndarray] = None
        self.actions: t.Optional[np.ndarray] = None
        self.next_states: t.Optional[np.ndarray] = None

        # track whether we are using image and the image formatter
        self._image_observations = image_observations
        self._image_formatter = image_formatter

        if replay_buffer is not None:
            # track the replay buffer
            self._replay_buffer = replay_buffer
        elif file_path is not None:
            print("Implement the load transitions dataset from disk method")
            import sys; sys.exit()
        else:
            raise NotImplementedError("You must specify either a replay buffer or file to load data from.")

        # get the length of each trajectory
        self.trajectory_lengths = [len(traj) for traj in replay_buffer.trajectories]

        self._device = device
        self._multi_gpu = multi_gpu

    def __len__(self) -> int:
        return len(self._replay_buffer)

    def __getitem__(self, indx: int) -> EnvironmentContrastiveDatapoint:
        """
        Return the specified sample from the dataset
        Args:
            indx: the index of inputs-target pair to be returned

        Returns:
            the environment transition inputs and the target bundled into a single
            datapoint object
        """
        # grab the transition at the given index from the replay buffer
        obs, action, _, next_obs, _, _, _, image_observation, next_image_observation = self._replay_buffer[indx]
        # check if our states are images or not
        if self._image_observations:
            state = image_observation
            target = next_image_observation
            if self._image_formatter is not None:
                # when not processing a batch of data, the image formatter adds a dimension at index 0
                # to create a batch of size 1. This does not work with our use of torch.stack() in the
                # collate method
                state = self._image_formatter(state).squeeze(0)
                target = self._image_formatter(target).squeeze(0)
        else:
            state = obs
            target = next_obs

        if self._target == "observation_difference":
            target = np.abs(np.subtract(target, state))

        # convert the numpy arrays to tensors
        states = torch.as_tensor(state)
        target = torch.as_tensor(target)
        actions = torch.as_tensor(action)

        return EnvironmentContrastiveDatapoint(state=states.float().to(self._device),
                                               action=actions.float().to(self._device),
                                               augmented_state=target.float().to(self._device))

    @property
    def observation_shape(self) -> t.Union[int, t.Sequence[int]]:
        if self._image_observations:
            sample_observation = self._replay_buffer.trajectories[0].initial_image_observations
            if self._image_formatter is not None:
                sample_observation = self._image_formatter(sample_observation, batch_states=True)
        else:
            sample_observation = self._replay_buffer.trajectories[0].initial_observations
        return sample_observation.shape[1:]

    @property
    def action_shape(self) -> t.Union[int, t.Sequence[int]]:
        sample_action = self._replay_buffer.trajectories[0].actions
        if len(sample_action.shape) == 2:
            return sample_action.shape[-1]
        else:
            # grab dimension sizes after the first and second dimensions to account for the dimensions holding the
            # trajectories and transitions
            return sample_action.shape[1:]

    def _flat_indx_to_trajectory_index(self, flat_indx: int) -> t.Tuple[int, int]:
        """
        Converts an index that assumes the transitions are flat to a trajectory and transition (w/in trajectory) index

        Args:
            flat_indx: the index assuming transitions are stored flat

        Returns:
            the index of the trajectory containing the transition
            the index of the transition within the trajectory
        """
        # need to figure out which transition indices are stored in which trajectories
        transition_cumulative_sum = np.cumsum(self.trajectory_lengths)
        # the trajectory containing the transition is at the first index where the cumulative sum of transitions is
        # less than the transition index
        target_trajectory_indx = int(np.argmax(flat_indx < transition_cumulative_sum))
        # get the transition's index within the trajectory as the different between the flat index and the cumulative
        # sum at the previous trajectory - tells us how far into the target trajectory the transition is
        if target_trajectory_indx == 0:
            transition_trajectory_indx = flat_indx
        else:
            transition_trajectory_indx = flat_indx - transition_cumulative_sum[target_trajectory_indx - 1]
        return target_trajectory_indx, transition_trajectory_indx

    @staticmethod
    def collate(batch: t.List[EnvironmentContrastiveDatapoint]) -> EnvironmentContrastiveBatch:
        """
        Collate a batch of environment transitions into a batch of environment transitions
        Args:
            batch: a list of environment transition datasets

        Returns:
            a batch of environment transitions
        """
        # used to accumulate the network inputs and targets
        states = []
        actions = []
        next_states = []

        # accumulate inputs and targets from each sample in the batch
        for sample in batch:
            states.append(sample.state)
            actions.append(sample.action)
            next_states.append(sample.augmented_state)

        # bundle the batch of inputs and the batch of targets into a single batch object
        # get item should already have put the tensor on the correct device
        return EnvironmentContrastiveBatch(states=torch.stack(states, dim=0),
                                           actions=torch.stack(actions, dim=0),
                                           augmented_states=torch.stack(next_states, dim=0))


class EnvironmentKStepTransitionDataset(EnvironmentTransitionDataset):
    """
    A dataset of environment transitions where the state-action pairs are inputs
    and the next states are the target values.

    The dataset can be loaded from a file saved to disk or from a Replay Buffer.

    States and next states cannot be images. The dataset is not correctly implemented to handle images.

    The dataset returns k transitions for each batch
    """
    def __init__(self, replay_buffer: t.Optional[TrajectoryReplayBuffer] = None,
                 file_path: t.Optional[Path] = None,
                 k: int = 1,
                 device: str = "cuda",
                 multi_gpu: bool = False,
                 image_observations: bool = False,
                 image_formatter: t.Optional[t.Any] = None):
        """
        Either the replay_buffer or the file_path needs to not be of type None. If neither are of type
        None then both are used to populate the dataset

        Args:
            replay_buffer: the buffer of collected environment transitions
            file_path: the location of the datasets file
            device: (default = cuda) whether to run on the cpu or a cuda device
            multi_gpu: (default = False) whether the model is trained across multiple GPUs in which case we do not
                       push the data to a device before returning it
            image_observations: (default = False) whether or not the states are to be tracked as images
            image_formatter: (default = None) a function to apply to the raw images in order to format them them for t
                             training
                TODO: define expected format for this file will probably do this
                      once get to the point of writing out a dataset file.
        """
        assert replay_buffer is not None or file_path is not None, ("One of replay_buffer or file_path must be "
                                                                    "specified. Both are None.")
        super(EnvironmentKStepTransitionDataset, self).__init__(replay_buffer=replay_buffer,
                                                                file_path=file_path,
                                                                device=device,
                                                                multi_gpu=multi_gpu,
                                                                image_observations=image_observations,
                                                                image_formatter=image_formatter)

        if replay_buffer is not None:
            self.not_dones = replay_buffer.all_not_dones
        self.k = k

    def __getitem__(self, indx: int) -> t.Optional[EnvironmentContrastiveDatapoint]:
        """
        Return the specified sample from the dataset
        Args:
            indx: the index of inputs-target pair to be returned

        Returns:
            the environment transition inputs and the target bundled into a single
            datapoint object
        """
        obs, actions, _, next_obs, not_dones, _, _, image_observations, next_image_observations = self._replay_buffer[tuple((indx, indx + self.k))]
        # check if our states are images or not
        if self._image_observations:
            states = image_observations
            targets = next_image_observations
            if self._image_formatter is not None:
                # when not processing a batch of data, the image formatter adds a dimension at index 0
                # to create a batch of size 1. This does not work with our use of torch.stack() in the
                # collate method
                states = self._image_formatter(states, batch_states=True)
                targets = self._image_formatter(targets, batch_states=True)
        else:
            states = obs
            targets = next_obs

        if self._target == "observation_difference":
            targets = np.abs(np.subtract(targets, states))

        # convert the numpy arrays to tensors
        states = torch.as_tensor(states)
        targets = torch.as_tensor(targets)
        actions = torch.as_tensor(actions)

        if not states.size()[0] == self.k:
            return None
        else:
            return EnvironmentContrastiveDatapoint(state=states.float().to(self._device),
                                                   action=actions.float().to(self._device),
                                                   augmented_state=targets.float().to(self._device))

    @staticmethod
    def collate(batch: t.List[EnvironmentContrastiveDatapoint]) -> EnvironmentContrastiveBatch:
        """
        Collate a batch of environment transitions into a batch of environment transitions
        Args:
            batch: a list of environment transition datasets

        Returns:
            a batch of environment transitions
        """
        # used to accumulate the network inputs and targets
        states = []
        actions = []
        augmented_states = []

        # accumulate inputs and targets from each sample in the batch
        for sample in batch:
            if sample is None: continue
            states.append(sample.state)
            actions.append(sample.action)
            augmented_states.append(sample.augmented_state)

        return EnvironmentContrastiveBatch(states=torch.stack(states, dim=0),
                                           actions=torch.stack(actions, dim=0),
                                           augmented_states=torch.stack(augmented_states, dim=0))
