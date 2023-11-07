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

from reed.data.environment_transition_dataset import EnvironmentContrastiveBatch

from reed.models.image_encoder import get_image_encoder
from reed.models.self_predictive_representations_model import StateActionSelfPredictiveRepresentationsNetworkEnsemble


class ImageStateConsistencyNetwork(nn.Module):
    def __init__(self,
                 state_size: t.List[int],
                 out_size: int = 1,
                 state_embed_size: int = 256,
                 hidden_size: int = 256,
                 ssl_state_encoder_mimics_reward_model: bool = True,
                 image_encoder_architecture: str = "pixl2r",
                 consistency_comparison_dim: int = 32,
                 consistency_projection_size: int = 128,
                 consistency_comparison_hidden_size: int = 256,
                 consistency_architecture: str = "mosaic",
                 with_consistency_prediction_head: bool = True,
                 image_hidden_num_channels: int = 32,
                 num_layers: int = 3):
        """
        Learns embeddings such that the representations of an image and an augmented image are consistent with
        one another in the latent space.

        Args:
            state_size: dimensionality of the states
            out_size: the size of the output
            state_embed_size: the size of the state's embedding
            hidden_size: the size of the hidden layer(s)
            ssl_state_encoder_mimics_reward_model: whether the state encoder mimics the reward model's
                                     architecture
            image_encoder_architecture: (default = "pixl2r") the architecture that is used for the image encoder
            consistency_comparison_dim: the number of dimensions to use when comparing the predicted augmented state
                                     representation and the actual augmented state representation
            consistency_projection_size: the number of hidden units the state representations are projected
            consistency_comparison_hidden_size: the number of dimensions to use when comparing the predicted
                                      augmented state representation and the actual augmented state
                                      representation
            consistency_architecture: (default = "mosaic") controls the architecture used to predict the augmented
                                      state representation and then to project the current and augmented state
                                      representations before comparing.The name of the architecture references
                                      the source paper. The options are "simsiam" and "mosaic"
            with_consistency_prediction_head: (default = True) whether to include a prediction head to
                                      prediction the target representation. When we train with SimCLR we do not
                                      use the prediction head
            image_hidden_num_channels: (default = 32) the number of channels in the hidden layers of the
                                      image encoder
            num_layers: the number of hidden layers
        """
        super(ImageStateConsistencyNetwork, self).__init__()
        assert image_encoder_architecture in {"pixl2r", "drqv2"}
        assert consistency_architecture in {"simsiam", "mosaic"}

        # track the dimensionality of the input, the output, and the hidden dimensions
        self._state_size = state_size
        self._out_size = out_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._image_encoder_architecture = image_encoder_architecture
        self._image_hidden_num_channels = image_hidden_num_channels

        self._state_embed_size = state_embed_size
        self._ssl_state_encoder_mimics_reward_model = ssl_state_encoder_mimics_reward_model
        self._consistency_projection_size = consistency_projection_size
        self._consistency_comparison_dim = consistency_comparison_dim
        self._consistency_comparison_hidden_size = consistency_comparison_hidden_size
        self._consistency_architecture = consistency_architecture
        self._with_consistency_prediction_head = with_consistency_prediction_head

        self._build()

    def _build_consistency_comparison_architecture(self) -> t.Tuple[nn.Module, nn.Module]:
        """
        Builds the network architecture used to project the current and augmented state representations and then predict
        the augmented state representation from the current state representation.
        """
        predictor = None
        if self._consistency_architecture == "simsiam":
            # architecture from the SimSiam code base
            # project the predicted and true augmented state representation
            projector = nn.Linear(256, self._consistency_projection_size)
            # build a 2-layer consistency predictor following:
            # https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py#L39
            if self._with_consistency_prediction_head:
                predictor = nn.Sequential(
                    nn.Linear(self._consistency_projection_size,
                              self._consistency_comparison_hidden_size,
                              bias=False),
                    nn.BatchNorm1d(self._consistency_comparison_hidden_size),
                    nn.ReLU(inplace=True),  # hidden layer
                    nn.Linear(self._consistency_comparison_hidden_size,
                              self._consistency_projection_size))  # output layer

        elif self._consistency_architecture == "mosaic":
            # project the predicted and true augmented state representation
            # from: https://github.com/rll-research/mosaic/blob/561814b40d33f853aeb93f1113a301508fd45274/mosaic/models/rep_modules.py#L63
            projector = nn.Sequential(
                # Rearrange('B T d H W -> (B T) d H W'),
                nn.BatchNorm1d(self._state_embed_size), nn.ReLU(inplace=True),
                # Rearrange('BT d H W -> BT (d H W)'),
                nn.Linear(self._state_embed_size, self._consistency_comparison_hidden_size), nn.ReLU(inplace=True),
                nn.Linear(self._consistency_comparison_hidden_size, self._consistency_projection_size),
                nn.LayerNorm(self._consistency_projection_size)
            )
            # from: https://github.com/rll-research/mosaic/blob/561814b40d33f853aeb93f1113a301508fd45274/mosaic/models/rep_modules.py#L118
            if self._with_consistency_prediction_head:
                predictor = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(self._consistency_projection_size, self._consistency_comparison_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(self._consistency_comparison_hidden_size, self._consistency_projection_size),
                    nn.LayerNorm(self._consistency_projection_size))

        else:
            raise NotImplementedError(f"{self._consistency_architecture} is not an implemented consistency "
                                      f"comparison architecture.")

        return projector, predictor

    def _build(self):
        """
        Build the 4 mini-networks that make up the model:
            state encoder

            state convolution encoder

            augmented state predictor

            augmented state projector
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
        state_encoder = []
        hidden_in_size = conv_out_size

        if self._ssl_state_encoder_mimics_reward_model:
            state_encoder.append(('state_dense1', nn.Linear(hidden_in_size, self._state_embed_size)))
            state_encoder.append(('state_leakyrelu1', nn.LeakyReLU(negative_slope=1e-2)))
        else:  # Mimics the state action encoder.
            for i in range(self._num_layers):
                state_encoder.append((f'state_dense{i + 1}', nn.Linear(hidden_in_size, self._hidden_size)))
                state_encoder.append((f'state_leakyrelu{i + 1}', nn.LeakyReLU(negative_slope=1e-2)))
                hidden_in_size = self._hidden_size
        self._state_encoder = nn.Sequential(OrderedDict(state_encoder))

        self._state_projector, self._consistency_predictor = self._build_consistency_comparison_architecture()

    def forward(self, transitions: torch.nn.ModuleDict) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the representation of the augmented state and encode the given augmented state

        Args:
            transitions: a dictionary containing a batch of environment transitions composed of states and augmented
            states. The keys must be: states and augmented_states. The states are images.
        Returns:
            predicted embedding of the augmented state - p in the SimSiam paper
            augmented state embedding (detached from the tensor graph) - z in the SimSiam paper
            dimensionality: (batch, time step)
        """
        # Encode the state and augmented state.
        states_embed = self._state_encoder(
            torch.flatten(self._state_conv_encoder(transitions["states"]),
                          start_dim=1))

        # predict and project the representation of the augmented state
        if self._with_consistency_prediction_head:
            augmented_state_pred = self._consistency_predictor(self._state_projector(states_embed))
        else:
            augmented_state_pred = self._state_projector(states_embed)
        # we don't want gradients to back-propagate into the learned parameters from anything we do with the
        # augmented state
        with torch.no_grad():
            # embed the augmented state
            augmented_state_embed = self._state_encoder(
                torch.flatten(self._state_conv_encoder(transitions["augmented_states"].contiguous()),
                              start_dim=1))
            # project the state embedding into a space where it can be compared with the predicted augmented state
            projected_augmented_state_embed = self._state_projector(augmented_state_embed)

        # from the SimSiam paper, this is p and z
        return augmented_state_pred, projected_augmented_state_embed

    def initialize_from_pretrained_net(self, pretrained_net: nn.Module, to_copy: t.Sequence[str]):
        """
        Initialize the self-supervised consistency model with the weights from the given pretrained net

        Only the parameters for which the two model share the same names are copied over.

        Args:
            pretrained_net: the network with the parameters that will be used to initialize the model weights
            to_copy: the name of the layers to copy from the pretrained network

        Returns:
            an initialized preference-based reward network
        """
        # create the state dictionary that will be used to initialize the self-supervised consistency network with the
        # given pre-trained network weights
        this_dict = self.state_dict()
        if isinstance(pretrained_net, nn.DataParallel):
            pretrained_dict = pretrained_net.module.state_dict()
        else:
            pretrained_dict = pretrained_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in this_dict and k in to_copy)}
        this_dict.update(pretrained_dict)

        # copy the weights over from the pre-trained net to the preference-based reward network
        self.load_state_dict(this_dict)


class StateConsistencyNetworkEnsemble(StateActionSelfPredictiveRepresentationsNetworkEnsemble):
    def __init__(self,
                 device: torch.device,
                 networks: t.Sequence[nn.Module]):
        """
        Ensemble of networks used to train state representations that are consistent with
        the network's encoding of an augmented version of the state.

        Args:
            device: which GPU or CPU device the network is to be run on
            networks: the networks that will make up the ensemble
        """
        super(StateConsistencyNetworkEnsemble, self).__init__(device=device, networks=networks)

    def forward(self,
                transitions: t.List[EnvironmentContrastiveBatch]) -> t.Tuple[
            t.Sequence[torch.Tensor], t.Sequence[torch.Tensor]]:
        """
        For each network, predict the representation of the augmented state and encode the given augmented state

        Args:
            transitions: a batch of environment transitions composed of states and augmented states
        Returns:
            predicted embedding of the augmented state - p in the SimSiam paper
            augmented state embedding (detached from the tensor graph) - z in the SimSiam paper
            dimensionality: (batch, time step)
        """
        augmented_state_preds = []
        projected_augmented_state_embeds = []
        for net_indx, net_batch in enumerate(transitions):
            net = self._ensemble[net_indx]
            # we need to convert the batch object to a dictionary in case we are using nn.DataParallel
            augmented_state_pred, projected_augmented_state_embed = net(attr.asdict(net_batch))
            augmented_state_preds.append(augmented_state_pred)
            projected_augmented_state_embeds.append(projected_augmented_state_embed)

        # from the SimSiam paper, this is p and z
        return augmented_state_preds, projected_augmented_state_embeds

    def save(self, model_dir: Path, env_id: str, step: int):
        """
        Save the ensemble to disk
        Args:
            model_dir: location to save the SSC nets
            env_id: the string identifier for the environment
            step: number of overall training steps taken before this save
        """
        for net_indx, net in enumerate(self._ensemble):
            torch.save(net.state_dict(), f'{model_dir.as_posix()}/{env_id}_ssl_consistency_model_{step}_{net_indx}.pt')
