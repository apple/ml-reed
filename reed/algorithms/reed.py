#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

#!/usr/bin/env python3
import typing as t

import numpy as np
import torch
from torch.utils.data import random_split

from reed.algorithms.pebble import PEBBLE

from reed.data.preprocess_images import PreProcessSFCTrain

from reed.models.reward_model import StateActionFusionRewardModel
from reed.models.self_predictive_representations_model import (
    ImageStateActionSelfPredictiveRepresentationsNetwork,
    StateActionSelfPredictiveRepresentationsNetwork, StateActionSelfPredictiveRepresentationsNetworkEnsemble,
    ImageStateActionKStepSelfPredictiveRepresentationsNetwork,
    StateActionKStepSelfPredictiveRepresentationsNetwork, StateActionKStepSelfPredictiveRepresentationsNetworkEnsemble)

from reed.algorithms.next_state_representation_consistency import (
    ConsistentNextStateRepresentationEnsembleTrainer, ConsistentKNextStateRepresentationEnsembleTrainer,
    ContrastiveConsistentNextStateRepresentationEnsembleTrainer,
    ContrastiveConsistentKNextStateRepresentationEnsembleTrainer)
from reed.data.environment_transition_dataset import (
    EnvironmentTransitionDataset, EnvironmentKStepTransitionDataset)
from reed.data.environment_transition_data_loader import EnvironmentTransitionEnsembleDataLoader


class OnlineREEDWorkspace(PEBBLE):
    def __init__(self, experiment_config):
        """
        Create the workspace that will be used to run the REED experiments.
        """

        super(OnlineREEDWorkspace, self).__init__(experiment_config=experiment_config)

        # used to track the number of updates to the reward model
        self._reward_model_update_counter = 0

        print("Creating the SFC ensemble")
        # create the ensemble of SPR networks
        self.sfc_ensemble = self.construct_sfc_ensemble()

        # determine which parameters the two networks have in common
        if isinstance(self.reward_model.ensemble[0], torch.nn.DataParallel):
            reward_state_dict = self.reward_model.ensemble[0].module.state_dict().keys()
        else:
            reward_state_dict = self.reward_model.ensemble[0].state_dict().keys()
        if isinstance(self.sfc_ensemble[0], torch.nn.DataParallel):
            sfc_state_dict = self.sfc_ensemble[0].module.state_dict().keys()
        else:
            sfc_state_dict = self.sfc_ensemble[0].state_dict().keys()
        self.shared_parameters = list(set(reward_state_dict).intersection(set(sfc_state_dict)))

        self._reward_model_train_accuracies = []
        self._sfc_train_loss = []
        self._sfc_valid_loss = []
        self._reward_ensemble_disagreement = []
        self._reward_ensemble_disagreement_before_pretrain = []
        self._reward_ensemble_disagreement_post_reward_update = []
        # used to track the number of times the reward model has been updated
        # determines when we updated the reward model according to the SFC objective if we are not always updating
        # on the SFC objective when updating on the reward model
        self._reward_model_update_counter = 0

    def construct_reward_ensemble(self) -> StateActionFusionRewardModel:
        """
        Create the reward ensemble as specified in the experiment config.
        """
        return StateActionFusionRewardModel(
            in_dim=self.reward_in_dim,
            obs_dim=self._reward_observation_dimensionality,
            action_dim=self.env.action_space.shape[0],
            state_embed_size=self.experiment_config.reward_state_embed_dim,
            action_embed_size=self.experiment_config.reward_action_embed_dim,
            hidden_embed_size=self.experiment_config.reward_hidden_embed_dim,
            num_layers=self.experiment_config.reward_num_hidden_layers,
            final_activation=self.experiment_config.activation,
            ensemble_size=self.experiment_config.ensemble_size,
            lr=self.experiment_config.reward_lr,
            optimizer=self.experiment_config.reward_optimizer,
            reward_train_batch=self.experiment_config.reward_train_batch,
            device=self.experiment_config.device,
            multi_gpu=self.multi_gpu,
            image_observations=self.experiment_config.reward_from_image_observations,
            image_encoder_architecture=self.experiment_config.image_encoder_architecture,
            image_hidden_num_channels=self.experiment_config.image_hidden_num_channels,
            grayscale_images=self.experiment_config.grayscale_images
        )

    def construct_sfc_ensemble(self) -> StateActionSelfPredictiveRepresentationsNetworkEnsemble:
        """
        Initialize the ensemble of self-future consistency models
        """
        # create the ensemble of self-future consistency models
        ensemble = []
        for _ in range(self.experiment_config.ensemble_size):
            # instantiating the self-future consistency model, trainer, and dataset objects for reward model pretraining
            if self.experiment_config.k_step_future_predictions > 1:
                if self.experiment_config.reward_from_image_observations:
                    self_future_consistency_model = ImageStateActionKStepSelfPredictiveRepresentationsNetwork(
                        state_size=self.reward_model.obs_dim,
                        action_size=self.reward_model.action_dim,
                        state_embed_size=self.reward_model.state_embed_size,
                        action_embed_size=self.reward_model.action_embed_size,
                        hidden_size=self.reward_model.hidden_embed_size,
                        image_encoder_architecture=self.reward_model.image_encoder_architecture,
                        image_hidden_num_channels=self.reward_model.image_hidden_num_channels,
                        consistency_projection_size=self.experiment_config.self_future_consistency_projection_size,
                        consistency_comparison_hidden_size=self.experiment_config.self_future_consistency_comparison_hidden_size,
                        consistency_architecture=self.experiment_config.self_future_consistency_architecture,
                        with_consistency_prediction_head=(not self.experiment_config.contrastive_sfc),
                        num_layers=self.reward_model.num_hidden_layers,
                        k=self.experiment_config.k_step_future_predictions,
                        with_batch_norm=self.experiment_config.with_batch_norm)
                else:
                    self_future_consistency_model = StateActionKStepSelfPredictiveRepresentationsNetwork(
                        state_size=self.reward_model.obs_dim,
                        action_size=self.reward_model.action_dim,
                        state_embed_size=self.reward_model.state_embed_size,
                        action_embed_size=self.reward_model.action_embed_size,
                        consistency_comparison_dim=self.experiment_config.consistency_comparison_dim,
                        hidden_size=self.reward_model.hidden_embed_size,
                        num_layers=self.reward_model.num_hidden_layers,
                        final_activation=self.reward_model.final_activation,
                        with_consistency_prediction_head=(not self.experiment_config.contrastive_sfc),
                        k=self.experiment_config.k_step_future_predictions,
                        with_batch_norm=self.experiment_config.with_batch_norm)
            else:
                if self.experiment_config.reward_from_image_observations:
                    self_future_consistency_model = ImageStateActionSelfPredictiveRepresentationsNetwork(
                        state_size=self.reward_model.obs_dim,
                        action_size=self.reward_model.action_dim,
                        state_embed_size=self.reward_model.state_embed_size,
                        action_embed_size=self.reward_model.action_embed_size,
                        hidden_size=self.reward_model.hidden_embed_size,
                        image_encoder_architecture=self.reward_model.image_encoder_architecture,
                        image_hidden_num_channels=self.reward_model.image_hidden_num_channels,
                        consistency_projection_size=self.experiment_config.self_future_consistency_projection_size,
                        consistency_comparison_hidden_size=self.experiment_config.self_future_consistency_comparison_hidden_size,
                        consistency_architecture=self.experiment_config.self_future_consistency_architecture,
                        with_consistency_prediction_head=(not self.experiment_config.contrastive_sfc),
                        num_layers=self.reward_model.num_hidden_layers)
                else:
                    self_future_consistency_model = StateActionSelfPredictiveRepresentationsNetwork(
                        state_size=self.reward_model.obs_dim,
                        action_size=self.reward_model.action_dim,
                        state_embed_size=self.reward_model.state_embed_size,
                        action_embed_size=self.reward_model.action_embed_size,
                        consistency_comparison_dim=self.experiment_config.self_future_consistency_comparison_hidden_size,
                        hidden_size=self.reward_model.hidden_embed_size,
                        num_layers=self.reward_model.num_hidden_layers,
                        final_activation=self.reward_model.final_activation,
                        with_consistency_prediction_head=(not self.experiment_config.contrastive_sfc))
            # check if the model will be run with Data Parallelism
            if self.multi_gpu:
                ensemble.append(torch.nn.DataParallel(self_future_consistency_model).to(self.device))
            else:
                ensemble.append(self_future_consistency_model.to(self.device))
        # convert the list of models to an ensemble
        if self.experiment_config.k_step_future_predictions > 1:
            ensemble = StateActionKStepSelfPredictiveRepresentationsNetworkEnsemble(self.device, ensemble)
        else:
            ensemble = StateActionSelfPredictiveRepresentationsNetworkEnsemble(self.device, ensemble)

        return ensemble

    def load_transition_dataset(self) -> t.Union[EnvironmentTransitionDataset, EnvironmentKStepTransitionDataset]:
        """
        Load the replay buffer data into a transition dataset

        Returns:
            populated environment transition dataset
        """
        # create the object that will be used to pre-process the observations for training
        observation_preprocessor = PreProcessSFCTrain(
            image_observations=self.experiment_config.reward_from_image_observations,
            grayscale_images=self.experiment_config.grayscale_images,
            normalize_images=self.experiment_config.normalized_images)
        # extract a transition dataset from the replay buffer
        if self.experiment_config.k_step_future_predictions > 1:
            return EnvironmentKStepTransitionDataset(
                replay_buffer=self.replay_buffer,
                k=self.experiment_config.k_step_future_predictions,
                device=self.experiment_config.device,
                multi_gpu=self.multi_gpu,
                image_observations=self.experiment_config.reward_from_image_observations,
                image_formatter=observation_preprocessor.format_state)
        else:
            return EnvironmentTransitionDataset(
                replay_buffer=self.replay_buffer,
                target=self.experiment_config.self_future_consistency_target,
                device=self.device,
                multi_gpu=self.multi_gpu,
                image_observations=self.experiment_config.reward_from_image_observations,
                image_formatter=observation_preprocessor.format_state)

    def train_reward_on_environment_dynamics(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Trains the SPR network on the environment transitions data.

        Instantiates a trainer for the SPR network which contains the actual train loop from the SPR network

        Once training is done, copy the parameters of the SPR network ensemble to the reward model

        Returns:
            train accuracy per epoch per ensemble member
            train loss per epoch per ensemble member
        """
        # load the shared parameters
        print('The shared layers are:', self.shared_parameters)
        for n_indx in range(self.experiment_config.ensemble_size):
            if isinstance(self.sfc_ensemble, torch.nn.DataParallel):
                self.sfc_ensemble.module[n_indx].initialize_from_pretrained_net(
                    self.reward_model.ensemble[n_indx],
                    to_copy=self.shared_parameters)
            else:
                if isinstance(self.sfc_ensemble[n_indx], torch.nn.DataParallel):
                    self.sfc_ensemble[n_indx].module.initialize_from_pretrained_net(
                        self.reward_model.ensemble[n_indx],
                        to_copy=self.shared_parameters)
                else:
                    self.sfc_ensemble[n_indx].initialize_from_pretrained_net(
                        self.reward_model.ensemble[n_indx],
                        to_copy=self.shared_parameters)

        if self.experiment_config.k_step_future_predictions > 1:
            if self.experiment_config.contrastive_sfc:
                # the trainer that will be used to train the self-future consistency model
                # trains with a contrastive loss
                trainer = ContrastiveConsistentKNextStateRepresentationEnsembleTrainer(
                    ensemble=self.sfc_ensemble,
                    learning_rate=self.experiment_config.self_future_consistency_lr,
                    optimizer=self.experiment_config.self_future_consistency_optimizer,
                    with_lr_scheduler=self.experiment_config.self_future_consistency_lr_scheduler,
                    batch_size=self.experiment_config.self_future_consistency_batch,
                    temperature=self.experiment_config.temperature)
            else:
                # the trainer that will be used to train the self-future consistency model
                trainer = ConsistentKNextStateRepresentationEnsembleTrainer(
                    ensemble=self.sfc_ensemble,
                    learning_rate=self.experiment_config.self_future_consistency_lr,
                    optimizer=self.experiment_config.self_future_consistency_optimizer,
                    with_lr_scheduler=self.experiment_config.self_future_consistency_lr_scheduler)
        else:
            if self.experiment_config.contrastive_sfc:
                # the trainer that will be used to train the self-future consistency model
                trainer = ContrastiveConsistentNextStateRepresentationEnsembleTrainer(
                    ensemble=self.sfc_ensemble,
                    learning_rate=self.experiment_config.self_future_consistency_lr,
                    optimizer=self.experiment_config.self_future_consistency_optimizer,
                    with_lr_scheduler=self.experiment_config.self_future_consistency_lr_scheduler,
                    batch_size=self.experiment_config.self_future_consistency_batch,
                    temperature=self.experiment_config.temperature,
                    log_wandb=self.log_wandb)
            else:
                # the trainer that will be used to train the self-future consistency model
                trainer = ConsistentNextStateRepresentationEnsembleTrainer(
                    ensemble=self.sfc_ensemble,
                    learning_rate=self.experiment_config.self_future_consistency_lr,
                    optimizer=self.experiment_config.self_future_consistency_optimizer,
                    with_lr_scheduler=self.experiment_config.self_future_consistency_lr_scheduler,
                    log_wandb=self.log_wandb)

        # create the transition dataset from the replay buffer
        transition_dataset = self.load_transition_dataset()

        # split the dataset into a train and a test split
        train_size = int(len(transition_dataset) * self.experiment_config.training_split_size)
        valid_size = len(transition_dataset) - train_size
        train_dataset, valid_dataset = random_split(transition_dataset, [train_size, valid_size])
        # create the train and valid data loaders
        train_data_loader = EnvironmentTransitionEnsembleDataLoader(
            train_dataset,
            batch_size=self.experiment_config.self_future_consistency_batch,
            shuffle=self.experiment_config.shuffle_ds,
            ensemble_size=self.experiment_config.ensemble_size,
            collate_fn=transition_dataset.collate)
        valid_data_loader = EnvironmentTransitionEnsembleDataLoader(
            valid_dataset,
            batch_size=self.experiment_config.self_future_consistency_batch,
            shuffle=self.experiment_config.shuffle_ds,
            ensemble_size=self.experiment_config.ensemble_size,
            collate_fn=transition_dataset.collate)
        # train the reward network on the dataset of environment transitions
        train_losses, valid_losses = trainer.train(train_loader=train_data_loader,
                                                   num_epochs=self.experiment_config.self_future_consistency_epochs,
                                                   valid_loader=valid_data_loader)

        # copy the updated parameters to the reward model
        self.reward_model.initialize_from_sfc_net(self.sfc_ensemble,
                                                  self.experiment_config.freeze_pretrained_parameters,
                                                  to_copy=self.shared_parameters)

        return train_losses, valid_losses

    def update_reward(self, first_flag: bool = False):
        """
        Update the preference dataset and train the reward model
        """
        # check if we update the reward model parameters with the REED objective during this reward model update
        if (self.experiment_config.sfc_train_interval == 1 or
                (self._reward_model_update_counter % self.experiment_config.sfc_train_interval == 0)):
            # update the state-action encoder(s) on the set of transitions the agent has experienced
            self.train_reward_on_environment_dynamics()

        # grow the preference dataset
        self.grow_preference_dataset(first_flag=first_flag)

        # train the reward model on the updated preference dataset
        train_accuracy = self.train_reward_on_preferences()

        # increase the counter tracking the number of times we have updated the reward model
        self._reward_model_update_counter += 1

        print(f"Reward function is updated!! ACC: {train_accuracy}")
