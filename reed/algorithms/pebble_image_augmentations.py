import typing as t

import numpy as np
import torch

from omegaconf import dictconfig, OmegaConf

from torch.utils.data import random_split

from reed.models.reward_model import StateActionFusionRewardModel
from reed.algorithms.pebble import PEBBLE
from reed.data.environment_transition_dataset import EnvironmentTransitionDataset
from reed.data.environment_observation_dataset import AugmentedEnvironmentObservationDataset
from reed.algorithms.next_state_representation_consistency import \
    ContrastiveConsistentNextStateRepresentationEnsembleTrainer, ConsistentNextStateRepresentationEnsembleTrainer
from reed.data.preprocess_images import PreProcessSFCTrain
from reed.data.environment_transition_data_loader import EnvironmentTransitionEnsembleDataLoader

from reed.models.self_supervised_consistency_model import ImageStateConsistencyNetwork, StateConsistencyNetworkEnsemble


class PEBBLEImageAugmentations(PEBBLE):
    def __init__(self, experiment_config: dictconfig.DictConfig):
        """
        Create the workspace that will be used to run the PEBBLE with an auxiliary image augmentation task experiments.
        """
        super(PEBBLEImageAugmentations, self).__init__(experiment_config=experiment_config)

        # used to track the number of updates to the reward model
        self._reward_model_update_counter = 0

        print("Creating the SSC ensemble")
        self.ssc_ensemble = self.construct_ssc_ensemble()

        # determine which parameters the two networks have in common
        if isinstance(self.reward_model.ensemble[0], torch.nn.DataParallel):
            reward_state_dict = self.reward_model.ensemble[0].module.state_dict().keys()
        else:
            reward_state_dict = self.reward_model.ensemble[0].state_dict().keys()
        if isinstance(self.ssc_ensemble[0], torch.nn.DataParallel):
            ssc_state_dict = self.ssc_ensemble[0].module.state_dict().keys()
        else:
            ssc_state_dict = self.ssc_ensemble[0].state_dict().keys()
        self.shared_parameters = list(set(reward_state_dict).intersection(set(ssc_state_dict)))

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
            image_encoder_architecture=self.experiment_config.image_encoder_architecture,
            image_hidden_num_channels=self.experiment_config.image_hidden_num_channels,
            image_observations=True,
            grayscale_images=self.experiment_config.grayscale_images
        )

    def construct_ssc_ensemble(self) -> StateConsistencyNetworkEnsemble:
        """
        Initialize the ensemble of self-supervised consistency models.
        """
        ensemble = []
        for _ in range(self.experiment_config.ensemble_size):
            # Instantiating the self-supervised consistency model, trainer, and dataset objects for reward model
            # pretraining.
            self_supervised_consistency_model = ImageStateConsistencyNetwork(
                state_size=self.reward_model.obs_dim,
                state_embed_size=self.reward_model.state_embed_size,
                hidden_size=self.reward_model.hidden_embed_size,
                ssl_state_encoder_mimics_reward_model=self.experiment_config.ssl_state_encoder_mimics_reward_model,
                image_encoder_architecture=self.experiment_config.image_encoder_architecture,
                consistency_comparison_dim=self.experiment_config.consistency_comparison_dim,
                consistency_projection_size=self.experiment_config.self_supervised_consistency_projection_size,
                consistency_comparison_hidden_size=self.experiment_config.self_supervised_consistency_comparison_hidden_size,
                consistency_architecture=self.experiment_config.self_supervised_consistency_architecture,
                with_consistency_prediction_head=(not self.experiment_config.contrastive_ssc),
                num_layers=self.reward_model.num_hidden_layers)

            # check if the model will be run with Data Parallelism
            if self.multi_gpu:
                ensemble.append(torch.nn.DataParallel(self_supervised_consistency_model).to(self.device))
            else:
                ensemble.append(self_supervised_consistency_model.to(self.device))

        # convert the list of models to an ensemble
        return StateConsistencyNetworkEnsemble(self.device, networks=ensemble)


    def load_transition_dataset(self) -> EnvironmentTransitionDataset:
        """
        Load the replay buffer data into a transition dataset

        Returns:
            populated environment observation dataset
        """
        # create the object that will be used to pre-process the observations for training
        observation_preprocessor = PreProcessSFCTrain(
            image_observations=self.experiment_config.reward_from_image_observations,
            grayscale_images=self.experiment_config.grayscale_images,
            normalize_images=self.experiment_config.normalized_images)

        return AugmentedEnvironmentObservationDataset(
            replay_buffer=self.replay_buffer,
            use_strong_augs=self.experiment_config.use_strong_augs,
            data_augs=self.experiment_config.data_augs,
            grayscale_all_images=self.experiment_config.grayscale_images,
            height=self.experiment_config.image_height,
            width=self.experiment_config.image_width,
            device=self.device,
            multi_gpu=self.multi_gpu,
            image_formatter=observation_preprocessor.format_state)

    def train_reward_on_image_augmentations(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Trains the SSC network on the environment transitions data.

        Instantiates a trainer for the SSC network which contains the actual train loop from the SSC network

        Once training is done, copy the parameters of the SSC network ensemble to the reward model

        Returns:
            train accuracy per epoch per ensemble member
            train loss per epoch per ensemble member
        """
        # load the shared parameters
        print('The shared layers are:', self.shared_parameters)

        for n_indx in range(self.experiment_config.ensemble_size):
            if isinstance(self.ssc_ensemble, torch.nn.DataParallel):
                self.ssc_ensemble.module[n_indx].initialize_from_pretrained_net(
                    self.reward_model.ensemble[n_indx],
                    to_copy=self.shared_parameters)
            else:
                if isinstance(self.ssc_ensemble[n_indx], torch.nn.DataParallel):
                    self.ssc_ensemble[n_indx].module.initialize_from_pretrained_net(
                        self.reward_model.ensemble[n_indx],
                        to_copy=self.shared_parameters)
                else:
                    self.ssc_ensemble[n_indx].initialize_from_pretrained_net(
                        self.reward_model.ensemble[n_indx],
                        to_copy=self.shared_parameters)

        if self.experiment_config.contrastive_ssc:
            # the trainer that will be used to train the self-supervised consistency model
            # trains with a contrastive loss
            trainer = ContrastiveConsistentNextStateRepresentationEnsembleTrainer(
                ensemble=self.ssc_ensemble,
                learning_rate=self.experiment_config.self_supervised_consistency_lr,
                optimizer=self.experiment_config.self_supervised_consistency_optimizer,
                with_lr_scheduler=self.experiment_config.self_supervised_consistency_lr_scheduler,
                batch_size=self.experiment_config.self_supervised_consistency_batch,
                temperature=self.experiment_config.temperature)
        else:
            trainer = ConsistentNextStateRepresentationEnsembleTrainer(
                ensemble=self.ssc_ensemble,
                learning_rate=self.experiment_config.self_supervised_consistency_lr,
                optimizer=self.experiment_config.self_supervised_consistency_optimizer,
                with_lr_scheduler=self.experiment_config.self_supervised_consistency_lr_scheduler)
        # create the transition dataset from the replay buffer
        transition_dataset = self.load_transition_dataset()
        # split the dataset into a train and a test split
        train_size = int(len(transition_dataset) * self.experiment_config.training_split_size)
        valid_size = len(transition_dataset) - train_size
        train_dataset, valid_dataset = random_split(transition_dataset, [train_size, valid_size])
        # create the train and valid data loaders
        train_data_loader = EnvironmentTransitionEnsembleDataLoader(
            train_dataset,
            batch_size=self.experiment_config.self_supervised_consistency_batch,
            shuffle=self.experiment_config.shuffle_ds,
            ensemble_size=self.experiment_config.ensemble_size,
            collate_fn=transition_dataset.collate)
        valid_data_loader = EnvironmentTransitionEnsembleDataLoader(
            valid_dataset,
            batch_size=self.experiment_config.self_supervised_consistency_batch,
            shuffle=self.experiment_config.shuffle_ds,
            ensemble_size=self.experiment_config.ensemble_size,
            collate_fn=transition_dataset.collate)
        # train the reward network on the dataset of environment transitions
        train_losses, valid_losses = trainer.train(train_loader=train_data_loader,
                                                   num_epochs=self.experiment_config.self_supervised_consistency_epochs,
                                                   valid_loader=valid_data_loader)

        # copy the updated parameters to the reward model
        self.reward_model.initialize_from_sfc_net(self.ssc_ensemble,
                                                  self.experiment_config.freeze_pretrained_parameters,
                                                  to_copy=self.shared_parameters)
        return train_losses, valid_losses

    def update_reward(self, first_flag: bool = False):
        """
        Update the preference dataset and train the reward model
        """
        # update the state-action encoder(s) on the set of transitions the agent has experienced
        self.train_reward_on_image_augmentations()

        # grow the preference dataset
        self.grow_preference_dataset(first_flag=first_flag)

        # train the reward model on the updated preference dataset
        train_accuracy = self.train_reward_on_preferences()

        print(f"Reward function is updated!! ACC: {train_accuracy}")
