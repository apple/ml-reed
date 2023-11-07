#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
"""
Functions and classes to preprocess the image inputs given to the reward, SSC, and SFC models.
"""

import numpy as np

from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Normalize, Grayscale


def _to_grayscale(observation: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale

    Args:
        observation: the state observations
    Returns:
        the state-action pairs as a single array
    """
    obs = observation.astype(float)
    obs[..., 0] *= 0.1140
    obs[..., 1] *= 0.587
    obs[..., 2] *= 0.2989
    return np.sum(obs, axis=-1, keepdims=True)


class ScaleZeroToOne:
    def __init__(self):
        self._scaler = 255.

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: the image whose channels should be scaled. The channel dimension should be the last dimension
        Returns:
            image with channels scaled 0 to 1
        """
        return np.divide(image, self._scaler)


class PreProcessInference:
    """
    Preprocess the data for inference by the reward, SSC, and SFC models
    """
    def __init__(self,
                 image_observations: bool = False,
                 grayscale_images: bool = True,
                 normalize_images: bool = True,
                 environment_id: str = "dmc"):
        """
        Args:
            image_observations: whether the observations are images
            grayscale_images: whether images observations should be in grayscale
            normalize_images: whether the image observations should be normalized
            environment_id: the environment from which the data is coming
        """
        self.image_observations = image_observations
        self.grayscale_images = grayscale_images
        self.normalize_images = normalize_images
        self.environment_id = environment_id

    @staticmethod
    def _channel_first_to_last(observation: np.ndarray,
                               batch_states: bool = False,
                               by_trajectory: bool = False) -> np.ndarray:
        """
        Move the channel from the first dimension to the last dimension
        """
        if batch_states and by_trajectory:
            return np.transpose(observation, (0, 1, 3, 4, 2))
        elif batch_states:
            return np.transpose(observation, (0, 2, 3, 1))
        else:
            return np.transpose(observation, (1, 2, 0))

    @staticmethod
    def _channel_last_to_first(observation: np.ndarray, batch_states: bool = False,
                               by_trajectory: bool = False) -> np.ndarray:
        """
        Move the channel from the last dimension to the first dimension
        Args:
            observation: the state observations
            batch_states: whether a batch of state is to be processed
            by_trajectory: whether the batch of states is structured by trajectory -> should only be
                           True when batch_sa=True
        Returns:
            the image with the channel dimension moved from first to last
        """
        # permute the input so that the channels are in the first dimension of the images
        if batch_states and by_trajectory:
            return np.transpose(observation, (0, 1, 4, 2, 3))
        elif batch_states:
            return np.transpose(observation, (0, 3, 1, 2))
        else:
            # permute the input so that the channels are in the first dimension
            obs = np.transpose(observation, (2, 0, 1))
            # add a dimension along the front for concatenation into the buffer
            return np.expand_dims(obs, axis=0)

    def format_state(self, obs: np.ndarray, batch_states: bool = False,
                     by_trajectory: bool = False, channel_first: bool = False) -> np.ndarray:
        """
        Args:
            obs: the state observations
            batch_states: whether a batch of state is to be processed
            by_trajectory: whether the batch of states is structured by trajectory -> should only be
                           True when batch_sa=True
            channel_first: whether the channel dimension is first when the observations are images.
        Returns:
            the state-action pairs as a single array
        """
        if self.image_observations:
            if channel_first:
                # move the channel dimension from first to last to avoid a bunch of logic in our formatting methods
                # that handles variable locations for the channel dimension
                obs = self._channel_first_to_last(observation=obs,
                                                  batch_states=batch_states,
                                                  by_trajectory=by_trajectory)
            if self.grayscale_images:
                obs = _to_grayscale(observation=obs)
            if self.normalize_images:
                # TODO: add normalization based on pixel mean and standard deviation instead of scaling 0 to 1
                obs = np.divide(obs, 255.)
            # move the channel dimension from first to last
            return self._channel_last_to_first(observation=obs, batch_states=batch_states, by_trajectory=by_trajectory)

        else:
            return obs.reshape(1, obs.shape[1:]) if batch_states else obs.reshape(1, obs.shape[0])

    def format_state_action(self, obs: np.ndarray, act: np.ndarray,
                            batch_sa: bool = False, by_trajectory: bool = False,
                            channel_first: bool = False) -> np.ndarray:
        """
        Args:
            obs: the state observations
            act: the actions associated with each state observation
            batch_sa: whether a batch of state-action pairs is to be processed
            by_trajectory: whether the batch of state-action pairs is structured by trajectory -> should only be
                           True when batch_sa=True
            channel_first: whether the channel dimension is first when the observations are images.
        Returns:
            the state-action pairs as a single array
        """
        if self.image_observations:
            if channel_first:
                # move the channel dimension from first to last to avoid a bunch of logic in our formatting methods
                # that handles variable locations for the channel dimension
                obs = self._channel_first_to_last(observation=obs,
                                                  batch_states=batch_sa,
                                                  by_trajectory=by_trajectory)
            if self.grayscale_images:
                obs = _to_grayscale(observation=obs)
            if self.normalize_images:
                # TODO: add normalization based on pixel mean and standard deviation instead of scaling 0 to 1
                obs = np.divide(obs, 255.)

            # get the dimensions of the image
            obs_dim = obs.shape[-3:]
            assert len(obs_dim) == 3
            # add the actions to the image channels and permute the input so that the channels are in the first
            # dimension of the images
            if batch_sa and by_trajectory:
                repeated_actions = np.tile(act.reshape((act.shape[0], act.shape[1], 1, 1, act.shape[-1])),
                                           (1, 1, obs_dim[0], obs_dim[1], 1))
            elif batch_sa:
                repeated_actions = np.tile(act.reshape((act.shape[0], 1, 1, act.shape[-1])),
                                           (1, obs_dim[0], obs_dim[1], 1))
            else:
                repeated_actions = np.tile(act.reshape((1, 1, -1)), (obs_dim[0], obs_dim[1], 1))
            sa_t = np.concatenate((obs, repeated_actions), axis=-1)
            return self._channel_last_to_first(sa_t, batch_states=batch_sa, by_trajectory=by_trajectory)
        else:
            sa_t = np.concatenate([obs, act], axis=-1)
            if batch_sa:
                return sa_t
            else:
                return sa_t.reshape(1, -1)


class PreProcessSFCTrain:
    """
    Preprocess the data for training the SFC model
    """
    def __init__(self,
                 image_observations: bool = False,
                 grayscale_images: bool = True,
                 normalize_images: bool = True,
                 environment_id: str = "dmc"):
        """
        Args:
            image_observations: whether the observations are images
            grayscale_images: whether images observations should be in grayscale
            normalize_images: whether the image observations should be normalized
            environment_id: the environment from which the data is coming
        """
        self.image_observations = image_observations
        self.grayscale_images = grayscale_images
        self.normalize_images = normalize_images
        self.environment_id = environment_id

    def format_state(self, obs: np.ndarray, batch_states: bool = False, by_trajectory: bool = False) -> np.ndarray:
        """
        Args:
            obs: the state observations
            batch_states: whether a batch of state is to be processed
            by_trajectory: whether the batch of states is structured by trajectory -> should only be
                           True when batch_sa=True
        Returns:
            the state-action pairs as a single array
        """
        if self.image_observations:
            if self.grayscale_images:
                obs = _to_grayscale(observation=obs)
            if self.normalize_images:
                # TODO: add normalization based on pixel mean and standard deviation instead of scaling 0 to 1
                obs = np.divide(obs, 255.)
            # permute the input so that the channels are in the first dimension of the images
            if batch_states and by_trajectory:
                return np.transpose(obs, (0, 1, 4, 2, 3))
            elif batch_states:
                return np.transpose(obs, (0, 3, 1, 2))
            else:
                # permute the input so that the channels are in the first dimension
                obs = np.transpose(obs, (2, 0, 1))
                # add a dimension along the front for concatenation into the buffer
                return obs.reshape(1, *obs.shape)
        else:
            return obs.reshape(1, obs.shape[1:]) if batch_states else obs.reshape(1, obs.shape[0])

    def format_state_action(self, obs: np.ndarray, act: np.ndarray,
                            batch_sa: bool = False, by_trajectory: bool = False) -> np.ndarray:
        """
        Args:
            obs: the state observations
            act: the actions associated with each state observation
            batch_sa: whether a batch of state-action pairs is to be processed
            by_trajectory: whether the batch of state-action pairs is structured by trajectory -> should only be
                           True when batch_sa=True
        Returns:
            the state-action pairs as a single array
        """
        if self.image_observations:
            if self.grayscale_images:
                obs = _to_grayscale(observation=obs)
            if self.normalize_images:
                # TODO: add normalization based on pixel mean and standard deviation instead of scaling 0 to 1
                obs = np.divide(obs, 255.)

            # get the dimensions of the image
            obs_dim = obs.shape[-3:]
            assert len(obs_dim) == 3
            # add the actions to the image channels and permute the input so that the channels are in the first
            # dimension of the images
            if batch_sa and by_trajectory:
                repeated_actions = np.tile(act.reshape((act.shape[0], act.shape[1], 1, 1, act.shape[-1])),
                                           (1, 1, obs_dim[0], obs_dim[1], 1))
                sa_t = np.concatenate((obs, repeated_actions), axis=-1)
                return np.transpose(sa_t, (0, 1, 4, 2, 3))
            elif batch_sa:
                repeated_actions = np.tile(act.reshape((act.shape[0], 1, 1, act.shape[-1])),
                                           (1, obs_dim[0], obs_dim[1], 1))
                sa_t = np.concatenate((obs, repeated_actions), axis=-1)
                return np.transpose(sa_t, (0, 3, 1, 2))
            else:
                repeated_actions = np.tile(act.reshape((1, 1, -1)), (obs_dim[0], obs_dim[1], 1))
                sa_t = np.concatenate((obs, repeated_actions), axis=-1)
                # permute the input so that the channels are in the first dimension
                sa_t = np.transpose(sa_t, (2, 0, 1))
                # add a dimension along the front for concatenation into the buffer
                return np.expand_dims(sa_t, axis=0)
        else:
            sa_t = np.concatenate([obs, act], axis=-1)
            if batch_sa:
                return sa_t
            else:
                return sa_t.reshape(1, -1)
