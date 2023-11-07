#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import typing as t
from pathlib import Path
from collections import OrderedDict
import attr

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from reed.data.environment_transition_dataset import EnvironmentContrastiveBatch

from reed.models.image_encoder import get_image_encoder


class ImageStateActionSelfPredictiveRepresentationsNetwork(nn.Module):
    def __init__(self,
                 state_size: t.List[int],
                 action_size: int,
                 out_size: int = 1,
                 state_embed_size: int = 256,
                 action_embed_size: int = 10,
                 hidden_size: int = 256,
                 image_encoder_architecture: str = "pixl2r",
                 image_hidden_num_channels: int = 32,
                 consistency_projection_size: int = 128,
                 consistency_comparison_hidden_size: int = 256,
                 consistency_architecture: str = "mosaic",
                 num_layers: int = 3,
                 with_consistency_prediction_head: bool = True,
                 with_batch_norm: bool = False):
        """
        Initial pass at a network used to train state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            state_size: dimensionality of the states
            action_size: dimensionality of the actions
            out_size: the size of the output
            hidden_size: the size of the hidden layer(s)
            image_encoder_architecture: (default = "pixl2r") the architecture that is used for the image encoder
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
            consistency_projection_size: the number of hidden units the state representations are projected
            consistency_comparison_hidden_size: the number of dimensions to use when comparing the predicted next state
                                                representation and the actual next state representation
            consistency_architecture: controls the architecture used to predict the next state representation and then
                                      to project the current and next state representations before comparing.
                                      The name of the architecture references the source paper.
                                      The options are "simsiam" and "mosaic"
            num_layers: the number of hidden layers
            with_consistency_prediction_head: (default = True) whether to include a prediction head to prediction the
                                              target representation. When we train with SimCLR we do not use the
                                              prediction head.
            with_batch_norm: (default = False) whether to use batch norm when training the SPR network
        """
        super(ImageStateActionSelfPredictiveRepresentationsNetwork, self).__init__()

        # track the dimensionality of the input, the output, and the hidden dimensions
        self._state_size = state_size
        self._action_size = action_size
        self._out_size = out_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._image_encoder_architecture = image_encoder_architecture
        assert image_encoder_architecture in {"pixl2r", "drqv2"}
        self._image_hidden_num_channels = image_hidden_num_channels

        self._state_embed_size = state_embed_size
        self._action_embed_size = action_embed_size
        self._consistency_projection_size = consistency_projection_size
        self._consistency_comparison_hidden_size = consistency_comparison_hidden_size
        self._consistency_architecture = consistency_architecture
        self._with_consistency_prediction_head = with_consistency_prediction_head
        self._with_batch_norm = with_batch_norm

        self._build()

    def _build_consistency_comparison_architecture(self) -> t.Tuple[nn.Module, t.Optional[nn.Module]]:
        """
        Builds the network architecture used to project the current and next state representations and then predict
        the next state representation from the current state representation.
        """
        # architecture from the SimSiam code base
        if self._consistency_architecture == "simsiam":
            # project the predicted and true next state representation
            projector = nn.Linear(self._state_embed_size, self._consistency_projection_size)

            if self._with_consistency_prediction_head:
                # build a 2-layer consistency predictor following:
                predictor = nn.Sequential(
                    nn.Linear(self._consistency_projection_size,
                              self._consistency_comparison_hidden_size,
                              bias=False),
                    nn.BatchNorm1d(self._consistency_comparison_hidden_size),
                    nn.ReLU(inplace=True),  # hidden layer
                    nn.Linear(self._consistency_comparison_hidden_size,
                              self._consistency_projection_size))  # output layer
            else:
                predictor = None
        elif self._consistency_architecture == "mosaic":
            # project the predicted and true next state representation
            # from: https://github.com/rll-research/mosaic/blob/561814b40d33f853aeb93f1113a301508fd45274/mosaic/models/rep_modules.py#L63
            projector = nn.Sequential(
                nn.BatchNorm1d(self._state_embed_size), nn.ReLU(inplace=True),
                nn.Linear(self._state_embed_size, self._consistency_comparison_hidden_size), nn.ReLU(inplace=True),
                nn.Linear(self._consistency_comparison_hidden_size, self._consistency_projection_size),
                nn.LayerNorm(self._consistency_projection_size)
            )
            if self._with_consistency_prediction_head:
                # from: https://github.com/rll-research/mosaic/blob/561814b40d33f853aeb93f1113a301508fd45274/mosaic/models/rep_modules.py#L118
                predictor = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(self._consistency_projection_size, self._consistency_comparison_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(self._consistency_comparison_hidden_size, self._consistency_projection_size),
                    nn.LayerNorm(self._consistency_projection_size))
            else:
                predictor = None
        else:
            raise NotImplementedError(f"{self._consistency_architecture} is not an implemented consistency "
                                      f"comparison architecture.")

        return projector, predictor

    def _build(self):
        """
        Build the 5 mini-networks that make up the model:
            state encoder
            action encoder
            state-action encoder

            next state predictor

            next state projector
        """
        # the observations are first encoded with a CNN and then projected to an embedding
        # space where they are combined with the action embedding
        self._state_conv_encoder = get_image_encoder(
            architecture=self._image_encoder_architecture,
            obs_dim=self._state_size, out_size=self._out_size,
            hidden_dim=self._hidden_size, hidden_depth=self._num_layers,
            image_hidden_num_channels=self._image_hidden_num_channels)
        # get the size of the output from the convnet
        conv_out_size = torch.flatten(self._state_conv_encoder(torch.rand(size=[1] + list(self._state_size)))).size()[0]
        # build the network that will encode the state features
        self._state_encoder = nn.Sequential(OrderedDict([
            ('state_dense1', nn.Linear(conv_out_size, self._state_embed_size)),
            ('state_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2)),
        ]))

        # build the network that will encode the action features
        self._action_encoder = nn.Sequential(OrderedDict([
            ('action_dense1', nn.Linear(self._action_size, self._action_embed_size)),
            ('action_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2))
        ]))

        # build the network that models the relationship between the state anc action embeddings
        state_action_encoder = []
        hidden_in_size = self._action_embed_size + self._state_embed_size
        for i in range(self._num_layers):
            state_action_encoder.append((f'trunk_dense{i+1}', nn.Linear(hidden_in_size, self._hidden_size)))
            state_action_encoder.append((f'trunk_leakyrelu{i+1}', nn.LeakyReLU(negative_slope=1e-2)))
            hidden_in_size = self._hidden_size
        if self._with_batch_norm:
            state_action_encoder.append((f'trunk_batchnorm{self._num_layers}', nn.BatchNorm1d(self._hidden_size)))
        self._state_action_encoder = nn.Sequential(OrderedDict(state_action_encoder))

        # this is a single dense layer because we want to focus as much of the useful semantic information as possible
        # in the state-action representation
        if self._with_batch_norm:
            self._next_state_predictor = nn.Sequential(nn.Linear(self._hidden_size, self._state_embed_size),
                                                       nn.BatchNorm1d(self._state_embed_size))
        else:
            self._next_state_predictor = nn.Linear(self._hidden_size, self._state_embed_size)

        self._next_state_projector, self._consistency_predictor = self._build_consistency_comparison_architecture()

    def forward(self, transitions: t.Mapping[str, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the representation of the next state and encode the given next state

        Args:
            transitions: a dictionary containing a batch of environment transitions composed of states, actions, and
                         next states. The keys must be: states, actions, and next_states. The states are images.
        Returns:
            predicted embedding of the next state - p in the SimSiam paper
            next state embedding (detached from the tensor graph) - z in the SimSiam paper
            dimensionality: (batch, time step)
        """
        # encode the state, the action, and the state-action pair
        states_embed = self._state_encoder(
            torch.flatten(self._state_conv_encoder(transitions["states"]),
                          start_dim=1))
        actions_embed = self._action_encoder(transitions["actions"])

        state_action_embed = self._state_action_encoder(torch.cat([states_embed, actions_embed], dim=-1))

        if self._with_consistency_prediction_head:
            # predict and project the representation of the next state
            next_state_pred = self._consistency_predictor(self._next_state_projector(self._next_state_predictor(state_action_embed)))
        else:
            # predict and project the representation of the next state
            next_state_pred = self._next_state_projector(self._next_state_predictor(state_action_embed))

        # we don't want gradients to back-propagate into the learned parameters from anything we do with the next state
        with torch.no_grad():
            # embed the next state
            next_state_embed = self._state_encoder(
                torch.flatten(self._state_conv_encoder(transitions["augmented_states"].contiguous()),
                              start_dim=1))
            # project the next state embedding into a space where it can be compared with the predicted next state
            projected_next_state_embed = self._next_state_projector(next_state_embed)

        # from the SimSiam paper, this is p and z
        return next_state_pred, projected_next_state_embed

    def initialize_from_pretrained_net(self, pretrained_net: t.Union[nn.Module, nn.DataParallel], to_copy: t.Sequence[str]):
        """
        Initialize the self-future consistency model with the weights from the given pretrained net

        Only the parameters for which the two model share the same names are copied over.

        Args:
            pretrained_net: the network with the parameters that will be used to initialize the model weights
            to_copy: the name of the layers to copy from the pretrained network

        Returns:
            an initialized preference-based reward network
        """
        # create the state dictionary that will be used to initialize the self-future consistency network with the
        # given pre-trained network weights
        this_dict = self.state_dict()
        # check if the pre-trained model is wrapped in a DataParallek
        if isinstance(pretrained_net, nn.DataParallel):
            pretrained_dict = pretrained_net.module.state_dict()
        else:
            pretrained_dict = pretrained_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in this_dict and k in to_copy)}
        this_dict.update(pretrained_dict)

        # copy the weights over from the pre-trained net to the preference-based reward network
        self.load_state_dict(this_dict)


class StateActionSelfPredictiveRepresentationsNetwork(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 out_size: int = 1,
                 state_embed_size: int = 64,
                 action_embed_size: int = 64,
                 hidden_size: int = 128,
                 consistency_comparison_dim: int = 32,
                 num_layers: int = 3,
                 final_activation: str = 'tanh',
                 subselect_features: t.Optional[t.List] = None,
                 with_consistency_prediction_head: bool = True):
        """
        Initial pass at a network used to train state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            state_size: dimensionality of the states
            action_size: dimensionality of the actions
            out_size: the size of the output
            hidden_size: the size of the hidden layer(s)
            consistency_comparison_dim: the number of dimensions to use when comparing the predicted next state
                                        representation and the actual next state representation
            num_layers: the number of hidden layers
            final_activation: the activation to use on the final layer
            subselect_features: (optional) when specified, the indices of the features to use for training the
                                reward net
            with_consistency_prediction_head: (default = True) whether to include a prediction head to prediction the
                                              target representation. When we train with SimCLR we do not use the
                                              prediction head.
        """
        super(StateActionSelfPredictiveRepresentationsNetwork, self).__init__()

        # track the dimensionality of the input, the output, and the hidden dimensions
        self._state_size = (state_size if subselect_features is None else len(subselect_features))
        self._action_size = action_size
        self._out_size = out_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._final_activation_type = final_activation

        self._state_embed_size = state_embed_size
        self._action_embed_size = action_embed_size
        self._consistency_comparison_dim = consistency_comparison_dim
        self._with_consistency_prediction_head = with_consistency_prediction_head

        self._subselect_features = subselect_features

        self._build()

    def _build(self):
        """
        Build the 5 mini-networks that make up the model:
            state encoder
            action encoder
            state-action encoder

            next state predictor

            next state projector
        """
        # build the network that will encode the state features
        self._state_encoder = nn.Sequential(OrderedDict([
            ('state_dense1', nn.Linear(self._state_size, self._state_embed_size)),
            ('state_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2))
        ]))

        # build the network that will encode the action features
        self._action_encoder = nn.Sequential(OrderedDict([
            ('action_dense1', nn.Linear(self._action_size, self._action_embed_size)),
            ('action_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2))
        ]))

        # build the network that models the relationship between the state anc action embeddings
        state_action_encoder = []
        hidden_in_size = self._action_embed_size + self._state_embed_size
        for i in range(self._num_layers):
            state_action_encoder.append((f'trunk_dense{i+1}', nn.Linear(hidden_in_size, self._hidden_size)))
            state_action_encoder.append((f'trunk_leakyrelu{i+1}', nn.LeakyReLU(negative_slope=1e-2)))
            hidden_in_size = self._hidden_size
        self._state_action_encoder = nn.Sequential(OrderedDict(state_action_encoder))

        # this is a single dense layer because we want to focus as much of the useful semantic information as possible
        # in the state-action representation
        self._next_state_predictor = nn.Linear(self._hidden_size, self._state_embed_size)

        self._next_state_projector = nn.Linear(self._state_embed_size, self._consistency_comparison_dim)

        if self._with_consistency_prediction_head:
            # build a 2-layer consistency predictor following:
            self._consistency_predictor = nn.Sequential(
                nn.Linear(self._consistency_comparison_dim,
                          (int(self._consistency_comparison_dim / 8)
                           if self._consistency_comparison_dim > 8 * 2
                           else self._consistency_comparison_dim),
                          bias=False),
                nn.BatchNorm1d((int(self._consistency_comparison_dim / 8)
                                if self._consistency_comparison_dim > 8 * 2
                                else self._consistency_comparison_dim)),
                nn.ReLU(inplace=True),  # hidden layer
                nn.Linear((int(self._consistency_comparison_dim / 8)
                           if self._consistency_comparison_dim > 8 * 2
                           else self._consistency_comparison_dim),
                          self._consistency_comparison_dim))  # output layer

    def forward(self, transitions: torch.nn.ModuleDict) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the representation of the next state and encode the given next state

        Args:
            transitions: a dictionary containing a batch of environment transitions composed of states, actions, and
                         next states. The keys must be: states, actions, and next_states.
        Returns:
            predicted embedding of the next state - p in the SimSiam paper
            next state embedding (detached from the tensor graph) - z in the SimSiam paper
            dimensionality: (batch, time step)
        """
        # encode the state, the action, and the state-action pair
        if self._subselect_features is not None:
            states_embed = self._state_encoder(transitions["states"][:, self._subselect_features])
        else:
            states_embed = self._state_encoder(transitions["states"])

        actions_embed = self._action_encoder(transitions["actions"])

        state_action_embed = self._state_action_encoder(torch.cat([states_embed, actions_embed], dim=-1))

        # predict and project the representation of the next state
        if self._with_consistency_prediction_head:
            next_state_pred = self._consistency_predictor(self._next_state_projector(self._next_state_predictor(state_action_embed)))
        else:
            next_state_pred = self._next_state_projector(self._next_state_predictor(state_action_embed))

        # we don't want gradients to back-propagate into the learned parameters from anything we do with the next state
        with torch.no_grad():
            # embed the next state
            next_state_embed = self._state_encoder(transitions["augmented_states"])
            # project the next state embedding into a space where it can be compared with the predicted next state
            projected_next_state_embed = self._next_state_projector(next_state_embed)

        # from the SimSiam paper, this is p and z
        return next_state_pred, projected_next_state_embed

    def initialize_from_pretrained_net(self, pretrained_net: t.Union[nn.Module, nn.DataParallel],
                                       to_copy: t.Sequence[str]):
        """
        Initialize the self-future consistency model with the weights from the given pretrained net

        Only the parameters for which the two model share the same names are copied over.

        Args:
            pretrained_net: the network with the parameters that will be used to initialize the model weights
            to_copy: the name of the layers to copy from the pretrained network

        Returns:
            an initialized preference-based reward network
        """
        # create the state dictionary that will be used to initialize the self-future consistency network with the
        # given pre-trained network weights
        this_dict = self.state_dict()
        # check if the pre-trained model is wrapped in a DataParallek
        if isinstance(pretrained_net, nn.DataParallel):
            pretrained_dict = pretrained_net.module.state_dict()
        else:
            pretrained_dict = pretrained_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in this_dict and k in to_copy)}
        this_dict.update(pretrained_dict)

        # copy the weights over from the pre-trained net to the preference-based reward network
        self.load_state_dict(this_dict)


class StateActionSelfPredictiveRepresentationsNetworkEnsemble(nn.Module):
    def __init__(self,
                 device: torch.device,
                 networks: t.Sequence[nn.Module]):
        """
        Initial pass at an ensemble of networks used to train state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            device: which GPU or CPU device the network is to be run on
            networks: the networks that will make up the ensemble
        """
        super(StateActionSelfPredictiveRepresentationsNetworkEnsemble, self).__init__()

        # convert the list of networks into a pytorch network list
        self._ensemble = nn.ModuleList(networks)

        # track the device
        self.device = device

    def __len__(self) -> int:
        """
        The number of networks in the ensemble
        """
        return len(self._ensemble)

    def __getitem__(self, item: int) -> nn.Module:
        return self._ensemble[item]

    def forward(self,
                transitions: t.List[EnvironmentContrastiveBatch]) -> t.Tuple[t.Sequence[torch.Tensor], t.Sequence[torch.Tensor]]:
        """
        For each network, predict the representation of the next state and encode the given next state

        Args:
            transitions: a batch of environment transitions composed of states, actions, and next states for each
                         network in the ensemble
        Returns:
            predicted embedding of the next state - p in the SimSiam paper
            next state embedding (detached from the tensor graph) - z in the SimSiam paper
            dimensionality: (batch, time step)
        """
        next_state_preds = []
        projected_next_state_embeds = []
        for net_indx, net_batch in enumerate(transitions):
            net = self._ensemble[net_indx]
            # we need to convert the batch object a dictionary in case we are using nn.DataParallel
            next_state_pred, projected_next_state_embed = net(attr.asdict(net_batch))
            next_state_preds.append(next_state_pred)
            projected_next_state_embeds.append(projected_next_state_embed)

        # from the SimSiam paper, this is p and z
        return next_state_preds, projected_next_state_embeds

    def save(self, model_dir: Path, env_id: str, step: int):
        """
        Save the ensemble to disk
        Args:
            model_dir: location to save the SFC nets
            env_id: the string identifier for the environment
            step: number of overall training steps taken before this save

        Returns:

        """
        for net_indx, net in enumerate(self._ensemble):
            torch.save(net.state_dict(), f'{model_dir.as_posix()}/{env_id}_sfc_model_{step}_{net_indx}.pt')


class ImageStateActionKStepSelfPredictiveRepresentationsNetwork(ImageStateActionSelfPredictiveRepresentationsNetwork):
    def __init__(self,
                 state_size: t.List[int],
                 action_size: int,
                 out_size: int = 1,
                 state_embed_size: int = 256,
                 action_embed_size: int = 10,
                 hidden_size: int = 256,
                 image_encoder_architecture: str = "pixl2r",
                 image_hidden_num_channels: int = 32,
                 consistency_projection_size: int = 128,
                 consistency_comparison_hidden_size: int = 256,
                 consistency_architecture: str = "mosaic",
                 num_layers: int = 3,
                 with_consistency_prediction_head: bool = True,
                 k: int = 1,
                 with_batch_norm: bool = False):
        """
        Initial pass at a network used to train state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            state_size: dimensionality of the states
            action_size: dimensionality of the actions
            out_size: the size of the output
            hidden_size: the size of the hidden layer(s)
            image_encoder_architecture: (default = "pixl2r") the architecture that is used for the image encoder
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the image encoder
            consistency_projection_size: the number of hidden units the state representations are projected
            consistency_comparison_hidden_size: the number of dimensions to use when comparing the predicted next state
                                                representation and the actual next state representation
            consistency_architecture: controls the architecture used to predict the next state representation and then
                                      to project the current and next state representations before comparing.
                                      The name of the architecture references the source paper.
                                      The options are "simsiam" and "mosaic"
            num_layers: the number of hidden layers
            with_consistency_prediction_head: (default = True) whether to include a prediction head to prediction the
                                              target representation. When we train with SimCLR we do not use the
                                              prediction head.
            k: (default=1) the number of steps into the future to predict
            with_batch_norm: (default = False) whether to use batch norm when training the SPR network
        """
        self.k = k
        super(ImageStateActionKStepSelfPredictiveRepresentationsNetwork, self).__init__(
            state_size=state_size,
            action_size=action_size,
            out_size=out_size,
            state_embed_size=state_embed_size,
            action_embed_size=action_embed_size,
            hidden_size=hidden_size,
            image_encoder_architecture=image_encoder_architecture,
            image_hidden_num_channels=image_hidden_num_channels,
            consistency_projection_size=consistency_projection_size,
            consistency_comparison_hidden_size=consistency_comparison_hidden_size,
            consistency_architecture=consistency_architecture,
            num_layers=num_layers,
            with_consistency_prediction_head=with_consistency_prediction_head,
            with_batch_norm=with_batch_norm
        )

    def forward_packed_sequence(self,
                                transitions: t.Mapping[str, PackedSequence]) -> t.Tuple[t.Sequence[torch.Tensor], t.Sequence[torch.Tensor]]:
        """
        Predict the representation of the next state and encode the given next state

        Args:
            transitions: a batch of environment transitions composed of states, actions, and next states
        Returns:
            sequence of k steps predicted embedding of the next state - p in the SimSiam paper
            sequence of k next state embedding (detached from the tensor graph) - z in the SimSiam paper
        """
        raise NotImplementedError("forward_packed_sequence() DOES NOT WORK, because packed sequence does not play "
                                  "nicely with DataParallel!!!!")
        # pull out the state, action, and next state features along with the batch sizes
        packed_states = transitions["states"].data
        state_batch_sizes = transitions["states"].batch_sizes
        packed_actions = transitions["actions"].data
        action_batch_sizes = transitions["actions"].batch_sizes
        packed_augmented_states = transitions["augmented_states"].data
        augmented_state_batch_sizes = transitions["augmented_states"].batch_sizes

        # accumulate the predicted latent next state and the true latent next state
        pred_next_state_embeds = []
        true_next_state_embeds = []

        # used to track the state embedding, because we only create a state embedding from the environment
        # observations once
        states_embed = None
        pred_next_states_embed = None
        # loop over the number of time steps - length of the list of batch sizes
        for step, batch_size in enumerate(state_batch_sizes):
            # the batch sizes should be identical
            assert action_batch_sizes[step] == state_batch_sizes[step] == augmented_state_batch_sizes[step]
            # get the start and end indices from the packed the state, action, and next state features
            start_indx = step * batch_size
            end_indx = start_indx + batch_size
            if states_embed is None:
                states_embed = self._state_encoder(
                    torch.flatten(self._state_conv_encoder(packed_states[start_indx:end_indx, :]),
                                  start_dim=1))
            else:
                # check if we loose a sample from the batch
                if pred_next_states_embed.size()[0] > batch_size:
                    states_embed = pred_next_states_embed[:batch_size, :]
                else:
                    states_embed = pred_next_states_embed

            actions_embed = self._action_encoder(packed_actions[start_indx:end_indx, :])

            state_action_embed = self._state_action_encoder(torch.cat([states_embed, actions_embed], dim=-1))
            # predict the latent representation of the next state
            pred_next_states_embed = self._next_state_predictor(state_action_embed)
            # project the latent next state prediction into a compressed state space and then optionally apply the
            # consistency comparison predictor
            if self._with_consistency_prediction_head:
                consist_next_state_pred = self._consistency_predictor(self._next_state_projector(pred_next_states_embed))
            else:
                consist_next_state_pred = self._next_state_projector(pred_next_states_embed)
            pred_next_state_embeds.append(consist_next_state_pred)
            # project the next state into its latent representation
            # we don't want gradients to back-propagate into the learned parameters from anything we do with the
            # next state
            with torch.no_grad():
                # embed the next state
                next_state_embed = self._state_encoder(
                    torch.flatten(self._state_conv_encoder(packed_augmented_states[start_indx:end_indx, :]),
                                  start_dim=1))
                # project the next state embedding into a space where it can be compared with the predicted next state
                projected_next_state_embed = self._next_state_projector(next_state_embed).detach()
                true_next_state_embeds.append(projected_next_state_embed)
                # print("projected_next_state_embed")
                # print(projected_next_state_embed)

        # from the SimSiam paper, this is p and z
        return pred_next_state_embeds, true_next_state_embeds

    def forward(self,
                transitions: t.Mapping[str, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the representation of the next state and encode the given next state

        The sequence of transitions must all be the same size

        Args:
            transitions: a batch of environment transitions composed of states, actions, and next states
        Returns:
            sequence of k steps predicted embedding of the next state - p in the SimSiam paper
            sequence of k next state embedding (detached from the tensor graph) - z in the SimSiam paper
        """
        # pull out the state, action, and next state features
        # the data have the format (batch_size, time_steps, features)
        states = transitions["states"]
        actions = transitions["actions"]
        augmented_states = transitions["augmented_states"]

        assert states.size()[1] == actions.size()[1] == augmented_states.size()[1]

        # accumulate the predicted latent next state and the true latent next state
        pred_next_state_embeds = []
        true_next_state_embeds = []

        # used to track the state embedding, because we only create a state embedding from the environment
        # observations once
        states_embed = None
        pred_next_states_embed = None
        # loop over the number of time steps - length of the list of batch sizes
        for step in range(states.size()[1]):
            # check if we need to create a state embedding or if we will use the previously predicted next state
            # embeddings
            if states_embed is None:
                # embed the current batch of states
                states_embed = self._state_encoder(torch.flatten(self._state_conv_encoder(states[:, step]),
                                                                 start_dim=1))
            else:
                # the previously predicted next state embedding becomes our current state embedding
                states_embed = pred_next_states_embed
            # embed the actions
            actions_embed = self._action_encoder(actions[:, step])
            # get the fused state-action embedding
            state_action_embed = self._state_action_encoder(torch.cat([states_embed, actions_embed], dim=-1))
            # predict the latent representation of the next state
            pred_next_states_embed = self._next_state_predictor(state_action_embed)
            # project the latent next state prediction into a compressed state space and then optionally apply the
            # consistency comparison predictor
            if self._with_consistency_prediction_head:
                consist_next_state_pred = self._consistency_predictor(self._next_state_projector(pred_next_states_embed))
            else:
                consist_next_state_pred = self._next_state_projector(pred_next_states_embed)
            pred_next_state_embeds.append(consist_next_state_pred.unsqueeze(dim=1))
            # project the next state into its latent representation
            # we don't want gradients to back-propagate into the learned parameters from anything we do with the
            # next state
            with torch.no_grad():
                # embed the next state
                next_state_embed = self._state_encoder(
                    torch.flatten(self._state_conv_encoder(augmented_states[:, step]),
                                  start_dim=1))
                # project the next state embedding into a space where it can be compared with the predicted next state
                projected_next_state_embed = self._next_state_projector(next_state_embed).detach()
                true_next_state_embeds.append(projected_next_state_embed.unsqueeze(dim=1))
        # from the SimSiam paper, this is p and z
        return torch.cat(pred_next_state_embeds, dim=1), torch.cat(true_next_state_embeds, dim=1)


class StateActionKStepSelfPredictiveRepresentationsNetwork(StateActionSelfPredictiveRepresentationsNetwork):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 out_size: int = 1,
                 state_embed_size: int = 64,
                 action_embed_size: int = 64,
                 hidden_size: int = 128,
                 consistency_comparison_dim: int = 32,
                 with_consistency_prediction_head: bool = True,
                 num_layers: int = 3,
                 final_activation: str = 'tanh',
                 subselect_features: t.Optional[t.List] = None,
                 k: int = 1,
                 with_batch_norm: bool = False):
        """
        Initial pass at a network used to train state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            state_size: dimensionality of the states
            action_size: dimensionality of the actions
            out_size: the size of the output
            hidden_size: the size of the hidden layer(s)
            consistency_comparison_dim: the number of dimensions to use when comparing the predicted next state
                                        representation and the actual next state representation
            with_consistency_prediction_head: (default = True) whether to include a prediction head to prediction the
                                              target representation. When we train with SimCLR we do not use the
                                              prediction head.
            num_layers: the number of hidden layers
            final_activation: the activation to use on the final layer
            subselect_features: (optional) when specified, the indices of the features to use for training the
                                reward net
            k: (default=1) the number of steps into the future to predict
            with_batch_norm: (default = False) whether or not to use batch norm when training the SPR network
        """
        self._with_batch_norm = with_batch_norm
        super(StateActionKStepSelfPredictiveRepresentationsNetwork, self).__init__(state_size,
                                                                                   action_size,
                                                                                   out_size,
                                                                                   state_embed_size,
                                                                                   action_embed_size,
                                                                                   hidden_size,
                                                                                   consistency_comparison_dim,
                                                                                   num_layers,
                                                                                   final_activation,
                                                                                   subselect_features,
                                                                                   with_consistency_prediction_head=with_consistency_prediction_head)

        self.k = k

    def _build(self):
        """
        Build the 5 mini-networks that make up the model:
            state encoder
            action encoder
            state-action encoder

            next state predictor

            next state projector
        """
        # build the network that will encode the state features
        self._state_encoder = nn.Sequential(OrderedDict([
            ('state_dense1', nn.Linear(self._state_size, self._state_embed_size)),
            ('state_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2))
            # ('state_tanh1', nn.Tanh())
        ]))

        # build the network that will encode the action features
        self._action_encoder = nn.Sequential(OrderedDict([
            ('action_dense1', nn.Linear(self._action_size, self._action_embed_size)),
            ('action_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2))
            # ('action_tanh1', nn.Tanh())
        ]))

        # build the network that models the relationship between the state anc action embeddings
        state_action_encoder = []
        hidden_in_size = self._action_embed_size + self._state_embed_size
        for i in range(self._num_layers):
            state_action_encoder.append((f'trunk_dense{i+1}', nn.Linear(hidden_in_size, self._hidden_size)))
            state_action_encoder.append((f'trunk_leakyrelu{i+1}', nn.LeakyReLU(negative_slope=1e-2)))
            hidden_in_size = self._hidden_size
        if self._with_batch_norm:
            state_action_encoder.append((f'trunk_batchnorm{self._num_layers}', nn.BatchNorm1d(self._hidden_size)))
        self._state_action_encoder = nn.Sequential(OrderedDict(state_action_encoder))
        # the forward model
        # this is a single dense layer because we want to focus as much of the useful semantic information as possible
        # in the state-action representation
        self._next_state_predictor = nn.Linear(self._hidden_size, self._state_embed_size, bias=False)
        if self._with_batch_norm:
            self._next_state_predictor_batch_norm = nn.BatchNorm1d(self._state_embed_size)
        # the function f from the SimSiam paper
        self._next_state_projector = nn.Linear(self._state_embed_size, self._consistency_comparison_dim, bias=False)
        if self._with_batch_norm:
            self._next_state_projector_batch_norm = nn.BatchNorm1d(self._consistency_comparison_dim)
        # the function h from the SimSiam paper
        self._consistency_predictor = nn.Linear(self._consistency_comparison_dim, self._consistency_comparison_dim)

    def forward(self,
                transitions: EnvironmentContrastiveBatch) -> t.Tuple[t.Sequence[torch.Tensor], t.Sequence[torch.Tensor]]:
        """
        Predict the representation of the next state and encode the given next state

        Args:
            transitions: a batch of environment transitions composed of states, actions, and next states
        Returns:
            sequence of k steps predicted embedding of the next state - p in the SimSiam paper
            sequence of k next state embedding (detached from the tensor graph) - z in the SimSiam paper
        """
        # pull out the state, action, and next state features along with the batch sizes
        packed_states = transitions.states.data
        state_batch_sizes = transitions.states.batch_sizes
        packed_actions = transitions.actions.data
        action_batch_sizes = transitions.actions.batch_sizes
        packed_next_states = transitions.augmented_states.data
        next_state_batch_sizes = transitions.augmented_states.batch_sizes

        # check if the state features need to be sub-selected
        if self._subselect_features is not None:
            packed_states = packed_states[:, self._subselect_features]
            packed_next_states = packed_next_states[:, self._subselect_features]

        # accumulate the predicted latent next state and the true latent next state
        pred_next_state_embeds = []
        true_next_state_embeds = []

        # used to track the state embedding, because we only create a state embedding from the environment
        # observations once
        states_embed = None
        pred_next_states_embed = None
        # loop over the number of time steps - length of the list of batch sizes
        for step, batch_size in enumerate(state_batch_sizes):
            # the batch sizes should be identical
            assert action_batch_sizes[step] == state_batch_sizes[step] == next_state_batch_sizes[step]
            # get the start and end indices from the packed the state, action, and next state features
            start_indx = step * batch_size
            end_indx = start_indx + batch_size
            if states_embed is None:
                states_embed = self._state_encoder(packed_states[start_indx:end_indx, :])
            else:
                # check if we loose a sample from the batch
                if pred_next_states_embed.size()[0] > batch_size:
                    states_embed = pred_next_states_embed[:batch_size, :]
                else:
                    states_embed = pred_next_states_embed

            actions_embed = self._action_encoder(packed_actions[start_indx:end_indx, :])
            
            state_action_embed = self._state_action_encoder(torch.cat([states_embed, actions_embed], dim=-1))
            # predict the latent representation of the next state
            pred_next_states_embed = self._next_state_predictor(state_action_embed)
            # project the latent next state prediction into a compressed state space and then apply the consistency
            # comparison predictor
            if self._with_batch_norm:
                pred_next_states_embed_batch_norm = self._next_state_predictor_batch_norm(pred_next_states_embed)
                consist_next_state_pred = self._consistency_predictor(
                    self._next_state_projector_batch_norm(self._next_state_projector(
                        pred_next_states_embed_batch_norm)))
            else:
                consist_next_state_pred = self._consistency_predictor(self._next_state_projector(pred_next_states_embed))

            pred_next_state_embeds.append(consist_next_state_pred)
            # project the next state into it's latent representation
            # we don't want gradients to back-propagate into the learned parameters from anything we do with the
            # next state
            with torch.no_grad():
                # embed the next state
                next_state_embed = self._state_encoder(packed_next_states[start_indx:end_indx, :])
                # project the next state embedding into a space where it can be compared with the predicted next state
                if self._with_batch_norm:
                    projected_next_state_embed = self._next_state_projector_batch_norm(
                        self._next_state_projector(next_state_embed)).detach()
                else:
                    projected_next_state_embed = self._next_state_projector(next_state_embed).detach()
                true_next_state_embeds.append(projected_next_state_embed)
                # print("projected_next_state_embed")
                # print(projected_next_state_embed)

        # from the SimSiam paper, this is p and z
        # return torch.vstack(pred_next_state_embeds), torch.vstack(true_next_state_embeds)
        return pred_next_state_embeds, true_next_state_embeds


class StateActionKStepSelfPredictiveRepresentationsNetworkEnsemble(StateActionSelfPredictiveRepresentationsNetworkEnsemble):
    def __init__(self, device, networks: t.Sequence[nn.Module]):
        """
        Initial pass at an ensemble of networks used to train state-action representations that are consistent with
        the network's encoding of the state that results from applying the given action in the given state

        Args:
            device: which GPU or CPU device to put the network on
            networks: the networks that will make up the ensemble
        """
        super(StateActionKStepSelfPredictiveRepresentationsNetworkEnsemble, self).__init__(device, networks)
