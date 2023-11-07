#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import typing as t

from abc import abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from reed.models.image_encoder import get_image_encoder
from reed.models.self_predictive_representations_model import StateActionSelfPredictiveRepresentationsNetworkEnsemble

from reed.data.preference_data_loader import PreferenceTripletEnsembleDataLoader


def _to_grayscale(img_obs: np.ndarray, batch_states: bool) -> np.ndarray:
    """
    Convert the RGB image observations to grayscale
    Args:
        img_obs: the batch of image observations to convert to grayscale
        batch_states: whether a batch of observations or a single observation is to be processed

    Returns:
        the grayscale batch os images
    """
    if batch_states:
        obs = img_obs.astype(float)
        obs[:, :, :, 0] *= 0.1140
        obs[:, :, :, 1] *= 0.587
        obs[:, :, :, 2] *= 0.2989
        return np.sum(obs, axis=-1, keepdims=True)
    else:
        obs = img_obs.astype(float)
        obs[:, :, 0] *= 0.1140
        obs[:, :, 1] *= 0.587
        obs[:, :, 2] *= 0.2989
        return np.sum(obs, axis=-1, keepdims=True)


class _BaseModel(nn.Module):
    """
    A base reward model
    """
    def __init__(self, in_dim: t.Union[t.List[int], int], out_size: int = 1,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 dropout_probability: float = 0.0,
                 train_with_dropout: bool = False):
        """
        A network to consume the state-based environment observations and actions

        Args:
            in_dim: dimensionality of the model's input
            out_size: the size of the output
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            final_activation: (default = tanh) the activation to use on the final layer
            dropout_probability: (default = 0.) probability with which to set a weight value to 0. during a forward pass
                                 a probability of 0, means no dropout
            train_with_dropout: whether to use the dropout layers at train time (if the dropout probability is
                                greater than 0.)
                                Another use for the dropout layers is at test time to assess model uncertainty.
        """
        super(_BaseModel, self).__init__()
        # track the dimensionality of the input, the output, and the hidden dimensions
        self._in_size = in_dim
        self._out_size = out_size
        self._hidden_size = hidden_dim
        self._num_layers = hidden_depth
        self._final_activation_type = final_activation

        self._dropout_prob = dropout_probability
        self._train_with_dropout = train_with_dropout
        self._dropout_enabled = dropout_probability > 0

        self._build()

    @abstractmethod
    def _build(self):
        """
        Build the network
        """
        pass

    @abstractmethod
    def _forward(self, states_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Assign a reward value to each transition in the trajectory

        Args:
            states_action_pairs: batch of states-action pairs
                    expected dimensionality: (batch, state_features+action_features)
            ** It is expected that indices tie the states and action together
        Returns:
            the predicted reward for the state-action pair(s)

        """
        pass

    def _enable_dropout(self):
        """ Function to enable the dropout layers, e.g. during test-time """
        for m in self.modules():
            if 'dropout' in m.__class__.__name__:
                print(m)
                m.train()

        self._dropout_enabled = True

    def _disable_dropout(self):
        """ Function to disable the dropout layers, e.g. during train time"""
        for m in self.modules():
            if 'dropout' in m.__class__.__name__:
                m.eval()

        self._dropout_enabled = False

    def forward(self, states_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Assign a reward value to each transition in the trajectory
        Args:
            states_action_pairs: batch of states-action pairs
                    expected dimensionality: (batch, state_features+action_features)
            ** It is expected that indices tie the states and action together
        Returns:
            the predicted reward for the state-action pair(s)
        """
        if self.training and not self._train_with_dropout and self._dropout_prob > 0:
            self._disable_dropout()

        return self._forward(states_action_pairs)

    def weight_decay_parameters(self) -> t.Tuple[t.Sequence, t.Sequence]:
        """
        Sort the model parameters by whether weight decay can be applied to them
        Returns:
            with weight decay params
            without weight decay params
        """
        # need to track which weights will have L2 penalty (weight decay) applied and which won't
        params_with_wd = []
        params_without_wd = []
        for m in self.modules():
            # we get the nested Modules in their nested structure
            # skip modules until we get the to leaf node modules
            if len(list(m.children())) > 0: continue
            if isinstance(m, nn.Linear):
                params_with_wd.append(m.weight)
                params_without_wd.append(m.bias)
            else:
                params_without_wd.extend(m.parameters())

        return params_with_wd, params_without_wd

    def from_pretrained(self, state_dict: t.OrderedDict[str, torch.Tensor]):
        """
        Load the given state dictionary to the model
        Args:
            state_dict: the state dictionary to load into memory

        Returns:

        """
        self.load_state_dict(state_dict)

    def estimate_uncertainty(self, states_action_pairs: torch.Tensor, num_samples: int = 100) -> np.ndarray:
        """
        Estimate model uncertainty over the given batch of data
        Args:
            states_action_pairs: batch of states-action pairs
                                 expected dimensionality: (batch, state_features+action_features)
                                 ** It is expected that indices tie the states and action together
            num_samples: the number of forward passes with different dropout configurations to run to estimate
                         the uncertainty
        Returns:
            variance over predictions across the different dropout configurations
        """
        with torch.no_grad():
            # check how dropout started, because we want to leave it how we started
            dropout_start_enabled = self._dropout_enabled
            if not dropout_start_enabled:
                self._enable_dropout()

            # estimate the predicted values num_samples many times
            repeat_estimates = []
            for _ in range(num_samples):
                estimate = self._forward(states_action_pairs).detach().cpu().numpy()
                repeat_estimates.append(estimate)
            if not dropout_start_enabled:
                self._disable_dropout()
        # combine the estimations
        estimates = np.concatenate(repeat_estimates, axis=-1)
        mean_estimation = np.mean(estimates, axis=-1, keepdims=True)
        return np.mean(np.square(np.subtract(mean_estimation, estimates)), axis=-1)

    def forward_with_dropout(self, states_action_pairs: torch.Tensor, num_samples: int = 100) -> np.ndarray:
        """
        Execute a forward pass of the given data with all but the dropout layers in eval mode
        Args:
            states_action_pairs: batch of states-action pairs
                                 expected dimensionality: (batch, state_features+action_features)
                                 ** It is expected that indices tie the states and action together
            num_samples: the number of forward passes with different dropout configurations to run to estimate
                         the uncertainty
        Returns:
            dropout predictions across the different dropout configurations
        """
        with torch.no_grad():
            # check how dropout started, because we want to leave it how we started
            dropout_start_enabled = self._dropout_enabled
            if not dropout_start_enabled:
                self._enable_dropout()

            # estimate the predicted values num_samples many times
            repeat_estimates = []
            for _ in range(num_samples):
                estimate = self._forward(states_action_pairs).detach().cpu().numpy()
                repeat_estimates.append(estimate)
            # combine the estimations
            estimates = np.hstack(repeat_estimates)
            if not dropout_start_enabled:
                self._disable_dropout()
        return estimates

    def random_init_head(self):
        """
        Set the final layers to be randomly initialized values
        """
        self._prediction_head.reset_parameters()


class StateActionNetwork(_BaseModel):
    def __init__(self, in_dim: int, out_size: int = 1,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 dropout_probability: float = 0.0,
                 train_with_dropout: bool = False):
        """
        A network to consume the state-based environment observations and actions

        Args:
            in_dim: dimensionality of the model's input
            out_size: the size of the output
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            final_activation: (default = tanh) the activation to use on the final layer
            dropout_probability: (default = 0.) probability with which to set a weight value to 0. during a forward pass
                                 a probability of 0, means no dropout
            train_with_dropout: whether to use the dropout layers at train time (if the dropout probability is
                                greater than 0.)
                                Another use for the dropout layers is at test time to assess model uncertainty.
        """
        super(StateActionNetwork, self).__init__(
            in_dim=in_dim,
            out_size=out_size,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            final_activation=final_activation,
            dropout_probability=dropout_probability,
            train_with_dropout=train_with_dropout
        )

    def _build(self):
        """
        Build the 4 mini-networks that make up the model:
            state encoder
            action encoder
            state-action encoder
            prediction head
        """
        # build the network that models the relationship between the state anc action embeddings
        network_body = []
        for i in range(self._num_layers):
            network_body.append((f'trunk_dense{i+1}', nn.Linear((self._hidden_size if i > 0 else self._in_size), self._hidden_size)))
            network_body.append((f'trunk_leakyrelu{i+1}', nn.LeakyReLU(negative_slope=1e-2)))
            network_body.append((f'trunk_dropout{i+1}', nn.Dropout(self._dropout_prob)))
        self._network_body = nn.Sequential(OrderedDict(network_body))

        # build the prediction head and select a final activation
        self._prediction_head = nn.Linear(self._hidden_size, self._out_size)
        if self._final_activation_type == 'tanh':
            self._final_activation = nn.Tanh()
        elif self._final_activation_type == 'sig':
            self._final_activation = nn.Sigmoid()
        else:
            self._final_activation = nn.ReLU()

    def _forward(self, states_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Assign a reward value to each transition in the trajectory

        Args:
            states_action_pairs: batch of states-action pairs
                    expected dimensionality: (batch, state_features+action_features)
            ** It is expected that indices tie the states and action together
        Returns:
            the predicted reward for the state-action pair(s)
        """
        state_action_embed = self._network_body(states_action_pairs)

        # predict the target values
        prediction = self._final_activation(self._prediction_head(state_action_embed))

        return prediction


class StateActionFusionNetwork(_BaseModel):
    def __init__(self, obs_dim: int, action_dim: int, out_size: int = 1,
                 obs_embed_dim: int = 64, action_embed_dim: int = 64,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 dropout_probability: float = 0.0,
                 train_with_dropout: bool = False):
        """
        Initial pass at a network used to train state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            obs_dim: dimensionality of the states
            action_dim: dimensionality of the actions
            out_size: the size of the output
            obs_embed_dim: the size of the state embedding
            action_embed_dim: the size of the action embedding
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            final_activation: the activation to use on the final layer
            dropout_probability: (default = 0.) probability with which to set a weight value to 0. during a forward pass
                                 a probability of 0, means no dropout
            train_with_dropout: whether to use the dropout layers at train time (if the dropout probability is
                                greater than 0.)
                                Another use for the dropout layers is at test time to assess model uncertainty.
        """
        self._action_size = action_dim
        self._state_embed_size = obs_embed_dim  # int(self._hidden_size/2)
        self._action_embed_size = action_embed_dim  # int(self._hidden_size/2)
        super(StateActionFusionNetwork, self).__init__(
            in_dim=obs_dim,
            out_size=out_size,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            final_activation=final_activation,
            dropout_probability=dropout_probability,
            train_with_dropout=train_with_dropout
        )

    def _build(self):
        """
        Build the 4 mini-networks that make up the model:
            state encoder
            action encoder
            state-action encoder
            prediction head
        """
        # build the network that will encode the state features
        self._state_encoder = nn.Sequential(OrderedDict([
            ('state_dense1', nn.Linear(self._in_size, self._state_embed_size)),
            ('state_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2)),
            ('state_dropout1', nn.Dropout(self._dropout_prob))
        ]))

        # build the netowrk that will encode the action features
        self._action_encoder = nn.Sequential(OrderedDict([
            ('action_dense1', nn.Linear(self._action_size, self._action_embed_size)),
            ('action_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2)),
            ('action_dropout1', nn.Dropout(self._dropout_prob))
        ]))

        # build the network that models the relationship between the state anc action embeddings
        state_action_encoder = []
        for i in range(self._num_layers):
            state_action_encoder.append((f'trunk_dense{i+1}', nn.Linear((self._hidden_size if i > 0 else self._state_embed_size + self._action_embed_size), self._hidden_size)))
            state_action_encoder.append((f'trunk_leakyrelu{i+1}', nn.LeakyReLU(negative_slope=1e-2)))
            state_action_encoder.append((f'trunk_dropout{i+1}', nn.Dropout(self._dropout_prob)))
        self._state_action_encoder = nn.Sequential(OrderedDict(state_action_encoder))

        # build the prediction head and select a final activation
        self._prediction_head = nn.Linear(self._hidden_size, self._out_size)
        if self._final_activation_type == 'tanh':
            self._final_activation = nn.Tanh()
        elif self._final_activation_type == 'sig':
            self._final_activation = nn.Sigmoid()
        else:
            self._final_activation = nn.ReLU()

    def _forward(self, states_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Assign a reward value to each transition in the trajectory

        Args:
            states_action_pairs: batch of states-action pairs
                    expected dimensionality: (batch, state_features+action_features)
            ** It is expected that indices tie the states and action together
        Returns:
            the predicted reward for the state-action pair(s)
        """
        # encode the state, the action, and the state-action pair
        if len(states_action_pairs.size()) == 1:
            states_embed = self._state_encoder(states_action_pairs[:self._in_size])
            actions_embed = self._action_encoder(states_action_pairs[-self._action_size:])
        elif len(states_action_pairs.size()) == 2:
            states_embed = self._state_encoder(states_action_pairs[:, :self._in_size])
            actions_embed = self._action_encoder(states_action_pairs[:, -self._action_size:])
        elif len(states_action_pairs.size()) == 3:
            states_embed = self._state_encoder(states_action_pairs[:, :, :self._in_size])
            actions_embed = self._action_encoder(states_action_pairs[:, :, -self._action_size:])
        else:
            raise NotImplementedError()

        state_action_embed = self._state_action_encoder(torch.cat([states_embed, actions_embed], dim=-1))

        # predict the target values
        prediction = self._final_activation(self._prediction_head(state_action_embed))

        return prediction


class ImageStateActionNetwork(_BaseModel):
    def __init__(self, obs_dim: t.List[int], out_size: int = 1,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 dropout_probability: float = 0.0,
                 train_with_dropout: bool = False,
                 image_encoder_architecture: str = "pixl2r",
                 image_hidden_num_channels: int = 32,
                 *kwargs):
        """
        Maps state-action pairs to some type of value where the state is an image

        Args:
            obs_dim: dimensionality of the state images (height, width, channels)
            out_size: the size of the output
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            final_activation: (default = tanh) the activation to use on the final layer
            dropout_probability: (default = 0.) probability with which to set a weight value to 0. during a forward pass
                                 a probability of 0, means no dropout
            train_with_dropout: whether to use the dropout layers at train time (if the dropout probability is
                                greater than 0.)
                                Another use for the dropout layers is at test time to assess model uncertainty.
            image_encoder_architecture: (default = "pixl2r") the architecture that is used for the image encoder
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
        """
        self._image_encoder_architecture = image_encoder_architecture
        assert image_encoder_architecture in {"pixl2r", "drqv2"}
        self._image_hidden_num_channels = image_hidden_num_channels

        super(ImageStateActionNetwork, self).__init__(
            in_dim=obs_dim,
            out_size=out_size,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            final_activation=final_activation,
            dropout_probability=dropout_probability,
            train_with_dropout=train_with_dropout
        )

    def _build(self):
        """
        """
        # build the image encoder
        self.convnet = get_image_encoder(
            architecture=self._image_encoder_architecture,
            obs_dim=self._in_size, out_size=self._out_size,
            hidden_dim=self._hidden_size, hidden_depth=self._num_layers,
            image_hidden_num_channels=self._image_hidden_num_channels)

        net = []

        # get the size of the output from the convnet
        in_size = torch.flatten(self.convnet(torch.rand(size=[1] + list(self._in_size)))).size()[0]
        for i in range(self._num_layers):
            net.append(nn.Linear(in_size, self._hidden_size))
            net.append(nn.LeakyReLU())
            net.append(nn.Dropout(self._dropout_prob))
            in_size = self._hidden_size
        net.append(nn.Linear(in_size, self._out_size))
        if self._final_activation_type == 'tanh':
            net.append(nn.Tanh())
        elif self._final_activation_type == 'sig':
            net.append(nn.Sigmoid())
        else:
            net.append(nn.ReLU())

        self._net = nn.Sequential(*net)

    def _forward(self, states_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Assign a reward value to each transition in the trajectory

        Args:
            states_action_pairs: batch of states-action pairs
                    expected dimensionality: (batch, state_features+action_features)
            ** It is expected that indices tie the states and action together
        Returns:
            the predicted reward for the state-action pair(s)
        """
        return self._net(torch.flatten(self.convnet(states_action_pairs), start_dim=1))


class ImageStateActionFusionNetwork(_BaseModel):
    def __init__(self, obs_dim: t.List[int], action_dim: int, out_size: int = 1,
                 obs_embed_dim: int = 64, action_embed_dim: int = 64,
                 hidden_dim: int = 128, hidden_depth: int = 3,
                 final_activation: str = 'tanh',
                 dropout_probability: float = 0.0,
                 train_with_dropout: bool = False,
                 image_encoder_architecture: str = "pixl2r",
                 image_hidden_num_channels: int = 32,
                 **kwargs):
        """
        Initial pass at a network used to train image state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            obs_dim: dimensionality of the states
            action_dim: dimensionality of the actions
            out_size: the size of the output
            obs_embed_dim: the size of the state embedding
            action_embed_dim: the size of the action embedding
            hidden_dim: the size of the hidden layer(s)
            hidden_depth: the number of hidden layers
            final_activation: the activation to use on the final layer
            dropout_probability: (default = 0.) probability with which to set a weight value to 0. during a forward pass
                                 a probability of 0, means no dropout
            train_with_dropout: whether to use the dropout layers at train time (if the dropout probability is
                                greater than 0.)
                                Another use for the dropout layers is at test time to assess model uncertainty.
            image_encoder_architecture: (default = "pixl2r") the architecture that is used for the image encoder
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
        """
        self._action_size = action_dim
        self._image_encoder_architecture = image_encoder_architecture
        assert image_encoder_architecture in {"pixl2r", "drqv2"}
        self._image_hidden_num_channels = image_hidden_num_channels

        self._state_embed_size = obs_embed_dim
        self._action_embed_size = action_embed_dim

        super(ImageStateActionFusionNetwork, self).__init__(
            in_dim=obs_dim,
            out_size=out_size,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            final_activation=final_activation,
            dropout_probability=dropout_probability,
            train_with_dropout=train_with_dropout
        )

    def _build(self):
        """
        Build the 4 mini-networks that make up the model:
            state encoder
            action encoder
            state-action encoder
            prediction head
        """
        # the observations are first encoded with a CNN and then projected to an embedding
        # space where they are combined with the action embedding
        self._state_conv_encoder = get_image_encoder(
            architecture=self._image_encoder_architecture,
            obs_dim=self._in_size, out_size=self._out_size,
            hidden_dim=self._hidden_size, hidden_depth=self._num_layers,
            image_hidden_num_channels=self._image_hidden_num_channels)

        # get the size of the output from the convnet
        conv_out_size = torch.flatten(self._state_conv_encoder(torch.rand(size=[1] + list(self._in_size)))).size()[0]
        # build the network that will encode the state features
        self._state_encoder = nn.Sequential(OrderedDict([
            ('state_dense1', nn.Linear(conv_out_size, self._state_embed_size)),
            ('state_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2)),
            ('state_dropout1', nn.Dropout(self._dropout_prob))
        ]))

        # build the network that will encode the action features
        self._action_encoder = nn.Sequential(OrderedDict([
            ('action_dense1', nn.Linear(self._action_size, self._action_embed_size)),
            ('action_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2)),
            ('action_dropout1', nn.Dropout(self._dropout_prob))
        ]))

        # build the network that models the relationship between the state anc action embeddings
        state_action_encoder = []
        for i in range(self._num_layers):
            state_action_encoder.append((f'trunk_dense{i + 1}', nn.Linear(
                (self._hidden_size if i > 0 else self._state_embed_size + self._action_embed_size), self._hidden_size)))
            state_action_encoder.append((f'trunk_leakyrelu{i + 1}', nn.LeakyReLU(negative_slope=1e-2)))
            state_action_encoder.append((f'trunk_dropout{i + 1}', nn.Dropout(self._dropout_prob)))
        self._state_action_encoder = nn.Sequential(OrderedDict(state_action_encoder))

        # build the prediction head and select a final activation
        self._prediction_head = nn.Linear(self._hidden_size, self._out_size)
        if self._final_activation_type == 'tanh':
            self._final_activation = nn.Tanh()
        elif self._final_activation_type == 'sig':
            self._final_activation = nn.Sigmoid()
        else:
            self._final_activation = nn.ReLU()

    def _forward(self, states_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Assign a reward value to each transition in the trajectory

        Args:
            states_action_pairs: batch of states-action pairs
                    expected dimensionality: (batch, state_features+action_features)
            ** It is expected that indices tie the states and action together
        Returns:
            the predicted reward for the state-action pair(s)
        """
        # encode the state, the action, and the state-action pair
        if len(states_action_pairs.shape) == 3:
            states_embed = self._state_encoder(
                torch.flatten(self._state_conv_encoder(states_action_pairs[0:self._in_size[0]]), start_dim=1))
            actions_embed = self._action_encoder(states_action_pairs[-self._action_size:, 0, 0])
        elif len(states_action_pairs.shape) == 4:
            states_embed = self._state_encoder(
                torch.flatten(self._state_conv_encoder(states_action_pairs[:, 0:self._in_size[0]]), start_dim=1))
            actions_embed = self._action_encoder(states_action_pairs[:, -self._action_size:, 0, 0])
        else:
            raise NotImplementedError()
        # state_action_embed = self._state_action_encoder(torch.concat([states_embed, actions_embed], dim=-1))
        state_action_embed = self._state_action_encoder(torch.cat([states_embed, actions_embed], dim=-1))
        # print(state_action_embed.shape)
        # predict the target values
        prediction = self._final_activation(self._prediction_head(state_action_embed))

        return prediction


class StateActionRewardModel:
    """
    Reward model that operates over state action pairs
    """
    def __init__(self,
                 in_dim: t.Union[int, t.List[int]],
                 ensemble_size: int = 3,
                 hidden_dim: int = 256,
                 hidden_layers: int = 3,
                 final_activation: str = 'tanh',
                 lr: float = 3e-4,
                 optimizer: str = "adam",
                 reward_train_batch: int = 128,
                 size_segment: int = 1,
                 device: torch.device = "cuda",
                 multi_gpu: bool = False,
                 image_observations: bool = False,
                 image_encoder_architecture: str = "pixl2r",
                 image_hidden_num_channels: int = 32,
                 grayscale_images: bool = True):
        # the device the model will be put on
        self.device = device
        # whether data parallelism should be used during model training
        self.multi_gpu = multi_gpu
        # reward model configuration
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.optimizer_type = optimizer
        self.ensemble = []
        self.paramlst = []
        self.optimizer = None
        self.model = None
        self.final_activation = final_activation
        self.size_segment = size_segment

        self.image_observations = image_observations
        self.image_encoder_architecture = image_encoder_architecture
        self.image_hidden_num_channels = image_hidden_num_channels
        self.grayscale_images = grayscale_images

        # construct the reward ensemble
        self.construct_ensemble()

        # parameters used to train the reward model on the preference labelled trajectories
        self.train_batch_size = reward_train_batch
        self.CEloss = nn.CrossEntropyLoss()

    def eval(self):
        """Set each reward model in the ensemble to evaluation mode"""
        self.ensemble = [net.eval() for net in self.ensemble]

    def train(self):
        """Set each reward model in the ensemble to train mode"""
        self.ensemble = [net.train() for net in self.ensemble]

    def softXEnt_loss(self, predicted: torch.Tensor, target: torch.Tensor):
        logprobs = F.log_softmax(predicted, dim=1)
        return -(target * logprobs).sum() / predicted.shape[0]

    def construct_ensemble(self):
        for _ in range(self.ensemble_size):
            if self.image_observations:
                model = ImageStateActionNetwork(self.in_dim,
                                                out_size=1,
                                                hidden_dim=self.hidden_dim,
                                                hidden_depth=self.hidden_layers,
                                                final_activation=self.final_activation,
                                                image_encoder_architecture=self.image_encoder_architecture,
                                                image_hidden_num_channels=self.image_hidden_num_channels).float()
            else:
                model = StateActionNetwork(self.in_dim,
                                           out_size=1,
                                           hidden_dim=self.hidden_dim,
                                           hidden_depth=self.hidden_layers,
                                           final_activation=self.final_activation).float()
            print(model)
            # check if the model will be run with Data Parallelism
            if self.multi_gpu:
                print(f"There are {torch.cuda.device_count()} GPU devices, so the reward ensemble WILL be trained "
                      f"using nn.DataParallel")
                self.ensemble.append(nn.DataParallel(model).to(self.device))
            else:
                print(f"There are {torch.cuda.device_count()} GPU devices, so the reward ensemble will NOT be trained "
                      f"using nn.DataParallel")
                self.ensemble.append(model.to(self.device))
            # track all model parameters
            self.paramlst.extend(model.parameters())
        # create a single optimizer applied to all ensemble members
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.paramlst, lr=self.lr)
        elif self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.paramlst, lr=self.lr)
        else:
            raise NotImplementedError(f"{self.optimizer_type} is not implemented as a reward optimizer and must be "
                                      f"one of 'adam' or 'sgd'.")

    def format_state(self, obs: np.ndarray, batch_states: bool = False, by_trajectory: bool = False):
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
            # check if the images needs to be converted to grayscale
            if self.grayscale_images:
                obs = _to_grayscale(obs, batch_states=batch_states)
            if batch_states:
                # permute the input so that the channels are in the first dimension
                if by_trajectory:
                    obs = np.transpose(obs, (0, 1, 4, 2, 3))
                else:
                    print(obs.shape)
                    obs = np.transpose(obs, (0, 3, 1, 2))
                return obs
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
            # check if the images needs to be converted to grayscale
            if self.grayscale_images:
                obs = _to_grayscale(obs, batch_states=batch_sa)
            if batch_sa:
                obs_dim = obs.shape[1:]
                # we concatenate the actions along channel dimension of the image
                if by_trajectory:
                    repeated_actions = np.tile(act.reshape((act.shape[0], act.shape[1], 1, 1, act.shape[-1])),
                                               (1, 1, obs_dim[0], obs_dim[1], 1))
                else:
                    repeated_actions = np.tile(act.reshape((act.shape[0], 1, 1, act.shape[-1])),
                                               (1, obs_dim[0], obs_dim[1], 1))
                # now concatenate the two
                sa_t = np.concatenate((obs, repeated_actions), axis=-1)
                # permute the input so that the channels are in the first dimension
                if by_trajectory:
                    sa_t = np.transpose(sa_t, (0, 1, 4, 2, 3))
                else:
                    sa_t = np.transpose(sa_t, (0, 3, 1, 2))
                return sa_t
            else:
                obs_dim = obs.shape
                # we concatenate the actions along channel dimension of the image
                repeated_actions = np.tile(act.reshape((1, 1, -1)), (obs_dim[0], obs_dim[1], 1))
                # now concatenate the two
                sa_t = np.concatenate((obs, repeated_actions), axis=-1)
                # permute the input so that the channels are in the first dimension
                sa_t = np.transpose(sa_t, (2, 0, 1))
                # add a dimension along the front for concatenation into the buffer
                return sa_t.reshape(1, *self.in_dim)
        else:
            sa_t = np.concatenate([obs, act], axis=-1)
            if batch_sa:
                return sa_t
            else:
                return sa_t.reshape(1, -1)

    def p_hat_member(self, x_1: np.ndarray, x_2: np.ndarray, member: int = -1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            # if we are using image observations, we need to collapse along the batch and time dimensions to push
            # a forward pass through the network
            # to compute the probabilities when then need to re-construct the batch and time dimensions
            if self.image_observations:
                # we need to compute the probabilities in batches to avoid out of memory issues
                # we use the train batch size as it should be an amount safe to put on the GPU's memory without causing
                # issues
                mb_size = self.train_batch_size
                start_indx = 0
                r_hat1 = None
                r_hat2 = None
                while start_indx < x_1.shape[0]:
                    # check if there is a mb_size worth of trajectories to still be processed
                    if start_indx + mb_size <= x_1.shape[0]:
                        mb_x_1 = x_1[start_indx:start_indx + mb_size].reshape((-1, *x_1.shape[2:]))
                        mb_x_2 = x_1[start_indx:start_indx + mb_size].reshape((-1, *x_1.shape[2:]))
                    else:
                        # process the leftover trajectories in a batch smaller than mb_size
                        mb_x_1 = x_1[start_indx:].reshape((-1, *x_1.shape[2:]))
                        mb_x_2 = x_2[start_indx:].reshape((-1, *x_2.shape[2:]))
                    # process the leftover trajectories in a batch smaller than mb_size
                    mb_rhat1 = self.r_hat_member(torch.from_numpy(mb_x_1).float().to(self.device),
                                                 member=member).detach().cpu().reshape((mb_size, x_1.shape[1], 1))
                    mb_rhat2 = self.r_hat_member(torch.from_numpy(mb_x_2).float().to(self.device),
                                                 member=member).detach().cpu().reshape((mb_size, x_2.shape[1], 1))
                    start_indx += mb_size

                    # accumulate the rhats
                    if r_hat1 is None:
                        r_hat1 = mb_rhat1
                        r_hat2 = mb_rhat2
                    else:
                        r_hat1 = torch.concat((r_hat1, mb_rhat1), dim=0)
                        r_hat2 = torch.concat((r_hat2, mb_rhat2))

            else:
                r_hat1 = self.r_hat_member(x_1, member=member).cpu()
                r_hat2 = self.r_hat_member(x_2, member=member).cpu()
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:, 0]

    def p_hat_entropy(self, x_1: np.ndarray, x_2: np.ndarray, member: int = -1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x: torch.Tensor, member: int = -1) -> torch.Tensor:
        # the network parameterizes r hat in eqn 1 from the paper
        # return self.ensemble[member](torch.from_numpy(x).float().to(device))
        return self.ensemble[member](x)

    def r_hat(self, x: np.ndarray):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the
        # rewards are already normalized and I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(torch.from_numpy(x).float().to(self.device), member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x: np.ndarray):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(torch.from_numpy(x).float().to(self.device), member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0)

    def save(self, model_dir: str, env_id: str, step: int):
        """
        Save the reward ensemble to disk

        Args:
            model_dir: path where the ensemble is to be saved
            env_id: the environment on which the ensemble has been trained
            step: the number of policy training steps
        """
        for member in range(self.ensemble_size):
            torch.save(
                self.ensemble[member].state_dict(), f'{model_dir}/{env_id}_reward_model_{step}_{member}.pt'
            )

    def train_reward(self,
                     preference_data_loader: PreferenceTripletEnsembleDataLoader,
                     num_epoch: int):
        """
        Train the reward model on the given preference dataset.

        Args:
            preference_data_loader: loads batches of preference triplets. Separated handles different preference
                                    dataset permutations for each member of the reward's ensemble.
            num_epoch: the number of training epochs to execute
        """
        # track the accuracy and loss by ensemble member per epoch
        ensemble_accuracies = np.zeros((num_epoch, self.ensemble_size))
        ensemble_losses = np.zeros((num_epoch, self.ensemble_size))

        # train the reward model for the specified number of epochs
        for epoch in range(num_epoch):
            if epoch % 10 == 0:
                print(f"Running preference training epoch {epoch} of {num_epoch}")
            epoch_ensemble_losses = np.zeros(self.ensemble_size)
            epoch_ensemble_acc = np.zeros(self.ensemble_size)
            # train on each batch
            for batch_indx, batch in enumerate(preference_data_loader):
                # confirm there is either a single batch to be shared by all networks in the reward ensemble or
                # a batch per network in the ensemble
                assert len(batch) == 1 or len(batch) == self.ensemble_size
                # we need to zero out the gradients before we begin to process this batch
                self.optimizer.zero_grad()
                # we will need to accumulate the loss across the ensemble members
                batch_loss = 0.0
                for member_indx, preference_triplet_batch in enumerate(batch):
                    # the predicted reward per transition in each trajectory
                    # check if we need to collapse the batch and time dimensions into one and then reconstruct the two
                    if self.image_observations:
                        # get the rewards for each transition in the trajectories one
                        traj_one_shape = preference_triplet_batch.trajectories_one.shape
                        formatted_trajectories_one = preference_triplet_batch.trajectories_one.reshape(
                            (-1, *traj_one_shape[2:]))
                        r_hat1 = self.r_hat_member(formatted_trajectories_one,
                                                   member=member_indx).reshape((traj_one_shape[0],
                                                                                traj_one_shape[1], 1))
                        # get the rewards for each transition in the trajectories two
                        traj_two_shape = preference_triplet_batch.trajectories_two.shape
                        formatted_trajectories_two = preference_triplet_batch.trajectories_two.reshape(
                            (-1, *traj_two_shape[2:]))
                        r_hat2 = self.r_hat_member(formatted_trajectories_two,
                                                   member=member_indx).reshape((traj_two_shape[0],
                                                                                traj_two_shape[1], 1))
                    else:
                        r_hat1 = self.r_hat_member(preference_triplet_batch.trajectories_one,
                                                   member=member_indx)
                        r_hat2 = self.r_hat_member(preference_triplet_batch.trajectories_two,
                                                   member=member_indx)
                    # compute the return per trajectory
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)

                    r_hat = torch.cat([r_hat1, r_hat2], dim=-1)

                    # compute the ensemble member's loss
                    curr_loss = self.CEloss(r_hat, preference_triplet_batch.preference_labels.squeeze())
                    # add the loss from the ensemble member to the batch loss
                    batch_loss += curr_loss
                    # track the loss for this ensemble member
                    epoch_ensemble_losses[member_indx] += curr_loss.item()

                    # compute the accuracy of the ensemble member's predictions
                    _, predicted = torch.max(r_hat.data, 1)
                    correct = (predicted == preference_triplet_batch.preference_labels.squeeze()).sum().item()
                    epoch_ensemble_acc[member_indx] += correct
                # compute the gradients
                batch_loss.backward()
                # apply the gradients to the model
                self.optimizer.step()
            # compute the ensemble accuracy for this epoch
            ensemble_accuracies[epoch] = epoch_ensemble_acc / preference_data_loader.dataset_length()
            # compute the mean ensemble loss for this epoch
            ensemble_losses[epoch] = epoch_ensemble_losses / preference_data_loader.dataset_length()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} mean accuracy = {np.mean(ensemble_accuracies[:epoch + 1]):.2f}")

            # check the current mean accuracy, if it is greater than 0.97 then terminate training
            if np.mean(ensemble_accuracies[epoch]) >= 0.97:
                print(f"Epoch accuracy {np.mean(ensemble_accuracies[epoch]):.2f} "
                      f"after {epoch} epochs triggered early stopping.")
                return ensemble_accuracies[:epoch + 1], ensemble_losses[:epoch + 1]

        print(f"Epoch {num_epoch} mean accuracy = {np.mean(ensemble_accuracies):.2f}")

        return ensemble_accuracies, ensemble_losses


class StateActionFusionRewardModel(StateActionRewardModel):
    """
    A preference-based reward network that separately encodes then fuses the state and action features
    """
    def __init__(self, in_dim: t.Union[int, t.List[int]],
                 obs_dim: t.Union[int, t.List[int]],
                 action_dim: t.Union[int, t.List[int]],
                 state_embed_size: int = 64, action_embed_size: int = 64,
                 hidden_embed_size: int = 256, num_layers: int = 3,
                 final_activation: str ='tanh', ensemble_size: int = 3,
                 lr: float = 3e-4, optimizer: str = "adam", weight_decay: float = 0,
                 reward_train_batch: int = 128, size_segment: int = 1,
                 device: torch.device = "cuda", multi_gpu: bool = False,
                 subselect_features: t.Optional[t.List] = None,
                 image_observations: bool = False,
                 image_encoder_architecture: str = "pixl2r",
                 image_hidden_num_channels: int = 32,
                 grayscale_images: bool = True):

        # track the dimensionality of the observations and the actions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # track the dimensionality of the observation and action embeddings
        self.observation_embed_dim = state_embed_size
        self.action_embed_dim = action_embed_size

        self.weight_decay = weight_decay
        self.subselect_features = subselect_features

        super(StateActionFusionRewardModel, self).__init__(in_dim=in_dim,
                                                           ensemble_size=ensemble_size,
                                                           hidden_dim=hidden_embed_size,
                                                           hidden_layers=num_layers,
                                                           lr=lr,
                                                           optimizer=optimizer,
                                                           final_activation=final_activation,
                                                           reward_train_batch=reward_train_batch,
                                                           size_segment=size_segment,
                                                           device=device,
                                                           multi_gpu=multi_gpu,
                                                           image_observations=image_observations,
                                                           image_encoder_architecture=image_encoder_architecture,
                                                           image_hidden_num_channels=image_hidden_num_channels,
                                                           grayscale_images=grayscale_images)

    @property
    def state_embed_size(self) -> int:
        return self.observation_embed_dim

    @property
    def action_embed_size(self) -> int:
        return self.action_embed_dim

    @property
    def hidden_embed_size(self) -> int:
        return self.hidden_dim

    @property
    def num_hidden_layers(self) -> int:
        return self.hidden_layers

    def initialize_from_sfc_net(self,
                                pretrained_ensemble: StateActionSelfPredictiveRepresentationsNetworkEnsemble,
                                freeze_pretrained_parameters: bool,
                                to_copy: t.Sequence[str]):
        """
        Initialize the reward models with the weights from the pretrained net
        Args:
            pretrained_ensemble: the ensemble of pretrained network weights - the structure of each pre-trained network
                                 must match that of the preference-based reward networks
            freeze_pretrained_parameters: whether to freeze the model parameters that were pre-trained
            to_copy: the name of the layers to copy from the pretrained network

        Returns:
            an initialized preference-based reward network
        """
        if isinstance(pretrained_ensemble, torch.nn.DataParallel):
            assert len(pretrained_ensemble.module) == len(self.ensemble), (
                "The pretrained ensemble must contain the same number "
                "of networks as the ensemble of preference-based "
                "reward models.")
        else:
            assert len(pretrained_ensemble) == len(self.ensemble), (
                "The pretrained ensemble must contain the same number "
                "of networks as the ensemble of preference-based "
                "reward models.")

        # copy the weights over to each network in the ensemble
        for n in range(len(self.ensemble)):
            # pull out the preference-based reward net and the pre-trained network
            if isinstance(self.ensemble[n], nn.DataParallel):
                pref_reward_net = self.ensemble[n].module
            else:
                pref_reward_net = self.ensemble[n]
            if isinstance(pretrained_ensemble, torch.nn.DataParallel):
                if isinstance(pretrained_ensemble.module[n], nn.DataParallel):
                    pretrained_net = pretrained_ensemble.module[n].module
                else:
                    pretrained_net = pretrained_ensemble.module[n]
            else:
                if isinstance(pretrained_ensemble[n], nn.DataParallel):
                    pretrained_net = pretrained_ensemble[n].module
                else:
                    pretrained_net = pretrained_ensemble[n]
            # create the state dictionary that will be used to initialize the preference-based reward network with the
            # pre-trained network weights
            pref_reward_dict = pref_reward_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_net.state_dict().items()
                               if (k in pref_reward_dict and k in to_copy)}
            pref_reward_dict.update(pretrained_dict)

            # copy the weights over from the pre-trained net to the preference-based reward network
            pref_reward_net.load_state_dict(pref_reward_dict)

            if freeze_pretrained_parameters:
                # we want to freeze each pre-trained layer in the preference-based reward net
                for name, param in pref_reward_net.named_parameters():
                    if name in to_copy:
                        param.requires_grad = False

    def construct_ensemble(self):
        # need to overwrite the ensemble that is created in the parent class so we don't end up with double
        # the ensembles
        self.ensemble = []

        # need to track which weights will have L2 penalty (weight decay) applied and which won't
        params_with_wd = []
        params_without_wd = []
        for member in range(self.ensemble_size):
            if self.image_observations:
                model = ImageStateActionFusionNetwork(
                    obs_dim=self.obs_dim, action_dim=self.action_dim, out_size=1,
                    obs_embed_dim=self.observation_embed_dim, action_embed_dim=self.action_embed_dim,
                    hidden_dim=self.hidden_dim, hidden_depth=self.hidden_layers,
                    final_activation=self.final_activation,
                    image_encoder_architecture=self.image_encoder_architecture).float()
            else:

                model = StateActionFusionNetwork(
                    obs_dim=self.obs_dim, action_dim=self.action_dim, out_size=1,
                    obs_embed_dim=self.observation_embed_dim, action_embed_dim=self.action_embed_dim,
                    hidden_dim=self.hidden_dim, hidden_depth=self.hidden_layers,
                    final_activation=self.final_activation, subselect_features=self.subselect_features)
            # check if the model will be run with Data Parallelism
            if self.multi_gpu:
                print(f"There are {torch.cuda.device_count()} GPU devices, so the reward ensemble WILL be trained "
                      f"using nn.DataParallel")
                self.ensemble.append(nn.DataParallel(model).to(self.device))
            else:
                print(
                    f"There are {torch.cuda.device_count()} GPU devices, so the reward ensemble will NOT be trained "
                    f"using nn.DataParallel")
                self.ensemble.append(model.to(self.device))
            # check which parameters should vs. should not have weight decay
            if self.weight_decay > 0:
                with_wd, without_wd = model.weight_decay_parameters()
                params_with_wd.extend(with_wd)
                params_without_wd.extend(without_wd)
            else:
                self.paramlst.extend(model.parameters())

        # create a single optimizer applied to all ensemble members
        if self.optimizer_type == "adam":
            if self.weight_decay > 0:
                self.optimizer = torch.optim.Adam([{"params": params_without_wd},
                                                   {"params": params_with_wd, "weight_decay": self.weight_decay}],
                                                  lr=self.lr, weight_decay=0)
            else:
                self.optimizer = torch.optim.Adam(self.paramlst, lr=self.lr)
        elif self.optimizer_type == "sgd":
            if self.weight_decay > 0:
                self.optimizer = torch.optim.SGD([{"params": params_without_wd},
                                                  {"params": params_with_wd, "weight_decay": self.weight_decay}],
                                                 lr=self.lr, weight_decay=0)
            else:
                self.optimizer = torch.optim.SGD(self.paramlst, lr=self.lr)
        else:
            raise NotImplementedError(f"{self.optimizer_type} is not implemented as a reward optimizer and must be "
                                      f"one of 'adam' or 'sgd'.")

    def format_state_action(self, obs: np.ndarray, act: np.ndarray,
                            batch_sa: bool = False, by_trajectory: bool = False):
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
            # check if the images needs to be converted to grayscale
            if self.grayscale_images:
                obs = _to_grayscale(obs, batch_states=batch_sa)
            if batch_sa:
                # we add channels to hold the actions, but then only place action values at
                # we have dimensions: batch_size, image height, image width, action dimensions
                actions = np.zeros((obs.shape[0], obs.shape[1],  obs.shape[2], self.action_dim), dtype=np.float32)
                # populate the action values
                actions[:, 0, 0, :] = act
                # now concatenate the two
                sa_t = np.concatenate((obs, actions), axis=-1)
                # permute the input so that the channels are in the first dimension
                if by_trajectory:
                    sa_t = np.transpose(sa_t, (0, 1, 4, 2, 3))
                else:
                    sa_t = np.transpose(sa_t, (0, 3, 1, 2))
                return sa_t
            else:
                # we add channels to hold the actions, but then only place action values at
                actions = np.zeros((obs.shape[0], obs.shape[1], self.action_dim), dtype=np.float32)
                # populate the action values
                actions[0, 0] = act
                # now concatenate the two
                sa_t = np.concatenate((obs, actions), axis=-1)
                # permute the input so that the channels are in the first dimension
                sa_t = np.transpose(sa_t, (2, 0, 1))
                # add a dimension along the front for concatenation into the buffer
                return sa_t.reshape(1, *self.in_dim)
        else:
            sa_t = np.concatenate([obs, act], axis=-1)
            if batch_sa:
                return sa_t
            else:
                return sa_t.reshape(1, -1)

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](x)

