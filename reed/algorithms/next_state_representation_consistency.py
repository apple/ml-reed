#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import typing as t

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reed.models.self_predictive_representations_model import (
    StateActionSelfPredictiveRepresentationsNetwork,
    StateActionSelfPredictiveRepresentationsNetworkEnsemble)
from reed.data.environment_transition_dataset import EnvironmentContrastiveBatch
from reed.data.environment_transition_data_loader import EnvironmentTransitionEnsembleDataLoader


class ConsistentNextStateRepresentationTrainer:
    """
    Learns a representation of state and (optionally) action pairs by predicting a representation for the
    next state that is consistent with the next state's actual representation.
    """
    def __init__(self,
                 model: StateActionSelfPredictiveRepresentationsNetwork,
                 learning_rate: float = 1e-5,
                 optimizer: str = "sgd"):
        """
        Args:
            model: the model state encodes state-action pairs and next states
            learning_rate: (default = 1e-5) step size per model update step
            optimizer: (default = sgd) which optimizer to use to train the model. Either SGD or Adam.
        """
        # track the encoder network
        self._model = model

        # create the optimizer
        if optimizer == "sgd":
            self._optim = torch.optim.SGD(self._model.parameters(), lr=learning_rate)
        elif optimizer == "adam":
            torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError(f"{optimizer} is not a valid optimizer for "
                                      f"ConsistentNextStateRepresentationTrainer")
        # create the loss
        if next(self._model.parameters()).is_cuda:
            self._loss = torch.nn.CosineSimilarity(dim=-1).cuda()
        else:
            self._loss = torch.nn.CosineSimilarity(dim=-1)

        self.batch_index = 0

    @property
    def network(self) -> torch.nn.Module:
        """
        Returns:
            the learned state-action pair representation network
        """
        return self._model

    def _shared_step(self,
                     inputs: EnvironmentContrastiveBatch) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute a forward pass through the network
        Args:
            inputs: a batch of environment transitions

        Returns:
            the predicted next state encoding
            the next state encoding (detached from the tensor graph by the model)
        """
        # predict the representation of the next state and encode the next state
        # the model returns both, so this method is really just a wrapper around the model's forward method
        return self._model(inputs)

    def _train_step(self, batch: EnvironmentContrastiveBatch) -> float:
        """
        Update the model on the given batch of data
        Args:
            batch: samples with which to update the model

        Returns:
            average batch loss
            batch accuracy
        """
        # need to zero out the gradients
        self._optim.zero_grad()

        # compute the predicted preference labels
        predicted_next_state_embed, next_state_embed = self._shared_step(batch)
        # compute cosine similarity on the batch of data and negate
        loss = - torch.mean(self._loss(predicted_next_state_embed, next_state_embed))
        # compute the gradients
        loss.backward()
        # apply the gradients to model parameters
        self._optim.step()

        return loss.item()

    def _train_epoch(
            self,
            train_loader: DataLoader) -> np.ndarray:
        """
        Executes a single training epoch
        Args:
            train_loader: the data loader that generates an epoch of batches from a EnvironmentTransitionDataset

        Returns:
            average loss on the batch of transitions
        """
        # set the model into train model
        self._model.train()

        # accumulate the training losses
        epoch_losses = []

        # update the model on each batch of data
        for batch in train_loader:
            # execute a training step on the batch
            batch_loss = self._train_step(batch)
            epoch_losses.append(batch_loss)

        return np.asarray(epoch_losses)

    def evaluate(
            self,
            test_loader: DataLoader) -> np.ndarray:
        """
        Evaluate the model on the given data
        Args:
            test_loader: data loader for data the model will be evaluated on
                         much generate batches from a EnvironmentTransitionDataset

        Returns:
            mean loss on the evaluation set
        """
        # set the model in evaluation mode to avoid gradients
        self._model.eval()

        # accumulate the training losses
        losses = []

        with torch.no_grad():
            # evaluate the model on each batch of data
            # we batch to fit in memory
            for batch in test_loader:
                # compute the predicted preference labels
                predicted_next_state_embed, next_state_embed = self._shared_step(batch)
                # compute mean cosine similarity on the batch of data and negate
                loss = - torch.mean(self._loss(predicted_next_state_embed, next_state_embed))
                # track the mean error
                losses.append(loss.item())

        return np.asarray(losses)

    def train(
            self,
            train_loader: DataLoader,
            num_epochs: int,
            valid_loader: t.Optional[DataLoader] = None) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Train the model's parameters for the specified number of epoch.

        The model is trained such that it's representation for the predicted next state is similar
        to the representations of the next state
        Args:
            train_loader: the data loader that generates batches from a EnvironmentTransitionDataset
            num_epochs: number of iterations over the dataset before halting training
            valid_loader: (optional) data loader for the EnvironmentTransitionDataset to evaluate model learning on during
                          training

        Returns:
            average train loss per epoch
            average valid loss per epoch
        """
        # track performance across epochs
        train_losses = []
        valid_losses = []

        # train and evaluate (optionally) for each epoch
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)

            # if a validation set has been given, evaluate on it
            if valid_loader is not None:
                valid_loss = self.evaluate(valid_loader)
                valid_losses.append(valid_loss)

        return np.asarray(train_losses), np.asarray(valid_losses)


class ConsistentNextStateRepresentationEnsembleTrainer:
    """
    Learns an ensemble of representations of state and (optionally) action pairs by predicting a representation for the
    next state that is consistent with the next state's actual representation.
    """
    def __init__(self,
                 ensemble: StateActionSelfPredictiveRepresentationsNetworkEnsemble,
                 learning_rate: float = 1e-5,
                 optimizer: str = "sgd",
                 with_lr_scheduler: bool = False):
        """
        Args:
            ensemble: the ensemble of models for state encodes state-action pairs and next states
            learning_rate: (default = 1e-5) step size per model update step
            optimizer: (default = sgd) which optimizer to use to train the model. Either SGD or Adam.
        """
        super().__init__()
        # track the encoder network
        self._ensemble = ensemble

        # create the optimizer
        assert optimizer.lower() in {"sgd", "adam"}, f"Optimizer must be one of 'sgd' or 'adam', not {optimizer}"
        if optimizer.lower() == "sgd":
            self._optim = torch.optim.SGD(self._ensemble.parameters(), lr=learning_rate)
        else:
            self._optim = torch.optim.Adam(self._ensemble.parameters(), lr=learning_rate)
        # check if we will use a learning rate scheduler
        if with_lr_scheduler:
            self._learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optim,
                                                                                       mode="min",
                                                                                       factor=0.5,
                                                                                       patience=50)
        else:
            self._learning_rate_scheduler = None
        # create the loss
        if next(self._ensemble.parameters()).is_cuda:
            self._loss = torch.nn.CosineSimilarity(dim=-1).cuda()
        else:
            self._loss = torch.nn.CosineSimilarity(dim=-1)

        self.batch_index = 0

        self._total_epochs = 0

    def _shared_step(self,
                     inputs: t.List[EnvironmentContrastiveBatch]) -> t.Tuple[t.Sequence[torch.Tensor], t.Sequence[torch.Tensor]]:
        """
        Execute a forward pass through the network
        Args:
            inputs: a batch of environment transitions

        Returns:
            the predicted next state encoding
            the next state encoding (detached from the tensor graph by the model)
        """
        # predict the representation of the next state and encode the next state
        # the model returns both, so this method is really just a wrapper around the model's forward method
        return self._ensemble(inputs)

    def _train_step(self, batch: t.List[EnvironmentContrastiveBatch]) -> t.Tuple[float, t.Optional[float]]:
        """
        Update the model on the given batch of data
        Args:
            batch: samples with which to update the model

        Returns:
            average batch loss
            (optionally) average accuracy on the batch of transitions
        """
        # need to zero out the gradients
        self._optim.zero_grad()

        # do the forward pass of predictions
        # compute the predicted preference labels
        predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

        loss = 0.
        # compute the loss for each model in the ensemble
        for (predicted_next_state_embed, next_state_embed) in zip(predicted_next_state_embeds, next_state_embeds):
            # compute cosine similarity on the batch of data and negate
            loss -= torch.mean(self._loss(predicted_next_state_embed, next_state_embed))
        # compute the gradients
        loss.backward()
        # apply the gradients to model parameters
        self._optim.step()
        if isinstance(self._ensemble, torch.nn.DataParallel):
            return loss.item() / len(self._ensemble.module), None
        else:
            return loss.item() / len(self._ensemble), None

    def _train_epoch(
            self,
            train_loader: EnvironmentTransitionEnsembleDataLoader) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        """
        Executes a single training epoch
        Args:
            train_loader: the data loader that generates an epoch of batches from a EnvironmentTransitionDataset

        Returns:
            average loss on the batch of transitions
            (optionally) average accuracy on the batch of transitions
        """
        # set the model into train model
        self._ensemble.train()

        # accumulate the training losses
        epoch_losses = []
        # accumulate the training accuracy, if given
        epoch_accuracies = []

        # update the model on each batch of data
        for batch_indx, batch in enumerate(train_loader):
            # execute a training step on the batch
            batch_loss, batch_accuracy = self._train_step(batch)
            epoch_losses.append(batch_loss)
            if batch_accuracy is not None:
                epoch_accuracies.append(batch_accuracy)

            if batch_indx % 100 == 0:
                print(f"Batch {batch_indx}: mean loss = {np.mean(epoch_losses)}; "
                      f"mean accuracy = {np.mean(epoch_accuracies)}")

        return np.asarray(epoch_losses), (np.asarray(epoch_accuracies) if len(epoch_accuracies) >= 1 else None)

    def _compute_state_encoder_small_weights_ratio(self, threshold: float = 1e-3) -> t.Mapping[str, float]:
        """
        Compute the ratio of weights in the state encoder that are below the given threshold

        Args:
            threshold: the value against which the state encoder weights are compared against
        Returns:
            mapping from ensemble member ID to the ratio of weights smaller than the threshold
        """
        ratio_per_net = {}
        for net_indx in range(len(self._ensemble)):
            # get the state encoder for this ensemble member
            if isinstance(self._ensemble._ensemble[net_indx], torch.nn.DataParallel):
                state_encoder_weights = self._ensemble._ensemble[net_indx].module._state_encoder.state_dict()['state_dense1.weight']
            else:
                state_encoder_weights = self._ensemble._ensemble[net_indx]._state_encoder.state_dict()[
                    'state_dense1.weight']
            # see how many weights are below the threshold
            below_threshold_count = torch.sum(torch.abs(state_encoder_weights) < threshold)
            # compute the ratio
            below_threshold_ratio = below_threshold_count / torch.numel(state_encoder_weights)
            ratio_per_net[f"sfc_state_encoder_weights<{threshold}_ratio/net_{net_indx}"] = below_threshold_ratio.item()
        return ratio_per_net

    def _compute_action_encoder_small_weights_ratio(self, threshold: float = 1e-3) -> t.Mapping[str, float]:
        """
        Compute the ratio of weights in the action encoder that are below the given threshold

        Args:
            threshold: the value against which the action encoder weights are compared against
        Returns:
            mapping from ensemble member ID to the ratio of weights smaller than the threshold
        """
        ratio_per_net = {}
        for net_indx in range(len(self._ensemble)):
            # get the state encoder for this ensemble member
            if isinstance(self._ensemble._ensemble[net_indx], torch.nn.DataParallel):
                action_encoder_weights = self._ensemble._ensemble[net_indx].module._action_encoder.state_dict()['action_dense1.weight']
            else:
                action_encoder_weights = self._ensemble._ensemble[net_indx]._action_encoder.state_dict()['action_dense1.weight']
            # see how many weights are below the threshold
            below_threshold_count = torch.sum(torch.abs(action_encoder_weights) < threshold)
            # compute the ratio
            below_threshold_ratio = below_threshold_count / torch.numel(action_encoder_weights)
            ratio_per_net[f"sfc_action_encoder_weights<{threshold}_ratio/net_{net_indx}"] = below_threshold_ratio.item()
        return ratio_per_net

    def evaluate(
            self,
            test_loader: EnvironmentTransitionEnsembleDataLoader) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        """
        Evaluate the model on the given data
        Args:
            test_loader: data loader for data the model will be evaluated on
                         much generate batches from a EnvironmentTransitionDataset

        Returns:
            mean loss on the evaluation set
            (optionally) mean accuracy on the evaluation set
        """
        # set the model in evaluation mode to avoid gradients
        self._ensemble.eval()

        # accumulate the training losses
        losses = []

        with torch.no_grad():
            # evaluate the model on each batch of data
            for batch in test_loader:
                # do the forward pass of predictions
                # compute the predicted preference labels
                predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

                # we need to initialize the loss to 0 so we can accumulate across networks
                loss = 0.
                # compute the loss for each model in the ensemble
                for (predicted_next_state_embed, next_state_embed) in zip(predicted_next_state_embeds, next_state_embeds):
                    # compute cosine similarity on the batch of data and negate
                    loss -= torch.mean(self._loss(predicted_next_state_embed.detach().cpu(),
                                                  next_state_embed.detach().cpu()))
                # track the mean error
                if isinstance(self._ensemble, torch.nn.DataParallel):
                    losses.append(loss.item() / len(self._ensemble.module))
                else:
                    losses.append(loss.item() / len(self._ensemble))

        return np.asarray(losses), None

    def train(
            self,
            train_loader: EnvironmentTransitionEnsembleDataLoader,
            num_epochs: int,
            valid_loader: t.Optional[EnvironmentTransitionEnsembleDataLoader] = None) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Train the model's parameters for the specified number of epoch.

        The model is trained such that it's representation for the predicted next state is similar
        to the representations of the next state
        Args:
            train_loader: the data loader that generates batches from a EnvironmentTransitionDataset
            num_epochs: number of iterations over the dataset before halting training
            valid_loader: (optional) data loader for the EnvironmentTransitionDataset to evaluate model learning on during
                          training

        Returns:
            average train loss per epoch
            average valid loss per epoch
        """
        # track performance across epochs
        train_losses = []
        valid_losses = []

        train_accuracies = []
        valid_accuracies = []

        # train and evaluate (optionally) for each epoch
        for epoch in range(num_epochs):
            self._total_epochs += 1
            train_loss, train_accuracy = self._train_epoch(train_loader)
            train_losses.append(train_loss)
            if train_accuracy is not None:
                train_accuracies.append(train_accuracy)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}")
                print("SFC train loss and accuracy", np.mean(train_losses), np.mean(train_accuracies))

            # if a validation set has been given, evaluate on it
            if valid_loader is not None:
                valid_loss, valid_accuracy = self.evaluate(valid_loader)
                # check if we are stepping the learning rate based on the validation loss
                if self._learning_rate_scheduler is not None:
                    self._learning_rate_scheduler.step(valid_loss)
                # track the validation results
                valid_losses.append(valid_loss)
                if valid_accuracy is not None:
                    valid_accuracies.append(valid_accuracy)
                if epoch % 10 == 0:
                    print("SFC validation loss and accuracy", np.mean(valid_losses), np.mean(valid_accuracies))

        return np.asarray(train_losses), np.asarray(valid_losses)


class ConsistentKNextStateRepresentationEnsembleTrainer(ConsistentNextStateRepresentationEnsembleTrainer):
    """
    Learns an ensemble of representations of state and (optionally) action pairs by predicting a representation for the
    kth next state that is consistent with the kth next state's actual representation.
    """
    def __init__(self,
                 ensemble: StateActionSelfPredictiveRepresentationsNetworkEnsemble,
                 learning_rate: float = 1e-5,
                 optimizer: str = "sgd",
                 with_lr_scheduler: bool = False):
        """
        Args:
            ensemble: the ensemble of models for state encodes state-action pairs and next states
            learning_rate: (default = 1e-5) step size per model update step
            optimizer: (default = sgd) which optimizer to use to train the model
            with_lr_scheduler: (default = False) whether to use the learning rate scheduler during training
        """
        super(ConsistentKNextStateRepresentationEnsembleTrainer, self).__init__(ensemble,
                                                                                learning_rate,
                                                                                optimizer,
                                                                                with_lr_scheduler=with_lr_scheduler)

    def _train_step(self, batch: t.List[EnvironmentContrastiveBatch]) -> t.Tuple[float, t.Optional[float]]:
        """
        Update the model on the given batch of data
        Args:
            batch: samples with which to update the model

        Returns:
            average batch loss
            batch accuracy
        """
        # need to zero out the gradients
        self._optim.zero_grad()

        # do the forward pass of predictions
        predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

        loss = 0.
        # compute the loss for each model in the ensemble
        for batch_indx, (predicted_next_state_embed, next_state_embed) in enumerate(zip(predicted_next_state_embeds, next_state_embeds)):
            # get the batch size and the number of time steps
            batch_size, time_steps, embed_size = predicted_next_state_embed.size()
            # compute the SimCLR loss on the batch of data and accumulate
            batch_loss = self._loss(
                predicted_next_state_embed.view((batch_size * time_steps, embed_size)),
                next_state_embed.view((batch_size * time_steps, embed_size)))
            loss -= torch.mean(batch_loss)
        # compute the gradients
        loss.backward()
        # apply the gradients to model parameters
        self._optim.step()

        return loss.item() / len(self._ensemble), None

    def evaluate(
            self,
            test_loader: EnvironmentTransitionEnsembleDataLoader) -> t.Tuple[np.ndarray, t.Optional[np.ndarray]]:
        """
        Evaluate the model on the given data
        Args:
            test_loader: data loader for data the model will be evaluated on
                         much generate batches from a EnvironmentTransitionDataset

        Returns:
            mean loss on the evaluation set
        """
        # set the model in evaluation mode to avoid gradients
        self._ensemble.eval()

        # accumulate the training losses
        losses = []

        with torch.no_grad():
            # evaluate the model on each batch of data
            for batch in test_loader:
                # do the forward pass of predictions
                # compute the predicted preference labels
                predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

                # we need to initialize the loss to 0 so we can accumulate across networks
                loss = 0.
                # compute the loss for each model in the ensemble
                for batch_indx, (predicted_next_state_embed, next_state_embed) in enumerate(zip(predicted_next_state_embeds, next_state_embeds)):
                    # get the batch size and the number of time steps
                    batch_size, time_steps, embed_size = predicted_next_state_embed.size()
                    # compute the SimCLR loss on the batch of data and accumulate
                    batch_loss = self._loss(
                        predicted_next_state_embed.view((batch_size * time_steps, embed_size)),
                        next_state_embed.view((batch_size * time_steps, embed_size)))
                    loss -= torch.mean(batch_loss)
                # track the mean error
                if isinstance(self._ensemble, torch.nn.DataParallel):
                    losses.append(loss.item() / len(self._ensemble.module))
                else:
                    losses.append(loss.item() / len(self._ensemble))

        return np.asarray(losses), None


class SimCLR_Loss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(SimCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        batch_size = z_i.size()[0]
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = torch.cat([z_i, z_j], dim=0)
        features = F.normalize(features, p=2, dim=-1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(positives.device)
        logits = logits / self.temperature
        loss = self.criterion(logits, labels)
        loss /= (batch_size * 2)

        # compute the number of instances for which the correct images were paired
        total_correct = torch.sum(torch.argmax(logits, dim=1) == labels)
        return loss, (total_correct / (z_i.size()[0] + z_j.size()[0])).item()


class ContrastiveConsistentNextStateRepresentationEnsembleTrainer(ConsistentNextStateRepresentationEnsembleTrainer):
    """
    Learns an ensemble of representations of state and (optionally) action pairs by predicting a representation for the
    next state that is consistent with the next state's actual representation.
    """
    def __init__(self,
                 ensemble: StateActionSelfPredictiveRepresentationsNetworkEnsemble,
                 learning_rate: float = 1e-5,
                 optimizer: str = "sgd",
                 with_lr_scheduler: bool = False,
                 batch_size: int = 256,
                 temperature: float = 0.5):
        """
        Args:
            ensemble: the ensemble of models for state encodes state-action pairs and next states
            learning_rate: (default = 1e-5) step size per model update step
            optimizer: (default = sgd) which optimizer to use to train the model
            batch_size: (default = 256) the size of the training batches
            temperature: (default = 0.5) the SimCLR temperature hyper-parameter
        """
        super(ContrastiveConsistentNextStateRepresentationEnsembleTrainer, self).__init__(ensemble,
                                                                                          learning_rate,
                                                                                          optimizer,
                                                                                          with_lr_scheduler=with_lr_scheduler)
        # track the encoder network
        self._ensemble = ensemble

        self._loss = SimCLR_Loss(batch_size, temperature)

    def _train_step(self, batch: t.List[EnvironmentContrastiveBatch]) -> t.Tuple[float, float]:
        """
        Update the model on the given batch of data
        Args:
            batch: samples with which to update the model

        Returns:
            average batch loss
            batch accuracy
        """
        # need to zero out the gradients
        self._optim.zero_grad()

        # do the forward pass of predictions
        # compute the predicted preference labels
        predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

        loss = 0.
        accuracy = 0.
        # compute the loss for each model in the ensemble
        for (predicted_next_state_embed, next_state_embed) in zip(predicted_next_state_embeds, next_state_embeds):
            # compute SimCLR loss on the batch of data - cross entropy loss
            ensemble_member_loss, ensemble_member_accuracy = self._loss(predicted_next_state_embed, next_state_embed)
            loss += ensemble_member_loss
            accuracy += ensemble_member_accuracy
        # compute the gradients
        loss.backward()
        # apply the gradients to model parameters
        self._optim.step()

        return loss.item() / len(self._ensemble), accuracy / len(self._ensemble)

    def evaluate(self, test_loader: DataLoader) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model on the given data
        Args:
            test_loader: data loader for data the model will be evaluated on
                         must generate batches from a EnvironmentTransitionDataset

        Returns:
            mean loss on the evaluation set
            mean accuracy on the evaluation set
        """
        # set the model in evaluation mode to avoid gradients
        self._ensemble.eval()

        # accumulate the evaluation losses and accuracies
        losses = []
        accuracies = []

        with torch.no_grad():
            # evaluate the model on each batch of data
            for batch in test_loader:
                # do the forward pass of predictions
                # compute the predicted preference labels
                predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

                # we need to initialize the loss to 0 so we can accumulate across networks
                loss = 0.
                accuracy = 0.
                # compute the loss for each model in the ensemble
                for (predicted_next_state_embed, next_state_embed) in zip(predicted_next_state_embeds, next_state_embeds):
                    # compute cosine similarity on the batch of data and negate
                    ensemble_member_loss, ensemble_member_accuracy = self._loss(
                        predicted_next_state_embed.detach().cpu(), next_state_embed.detach().cpu())
                    loss += ensemble_member_loss
                    accuracy += ensemble_member_accuracy
                # track the mean error
                losses.append(loss.item() / len(self._ensemble))
                # track the mean accuracy
                accuracies.append(accuracy / len(self._ensemble))

        return np.asarray(losses), np.asarray(accuracies)


class ContrastiveConsistentKNextStateRepresentationEnsembleTrainer(ConsistentNextStateRepresentationEnsembleTrainer):
    """
    Learns an ensemble of representations of state and (optionally) action pairs by predicting a representation for the
    kth next state that is consistent with the kth next state's actual representation.
    """
    def __init__(self,
                 ensemble: StateActionSelfPredictiveRepresentationsNetworkEnsemble,
                 learning_rate: float = 1e-5,
                 optimizer: str = "sgd",
                 with_lr_scheduler: bool = False,
                 batch_size: int = 256,
                 temperature: float = 0.5):
        """
        Args:
            ensemble: the ensemble of models for state encodes state-action pairs and next states
            learning_rate: (default = 1e-5) step size per model update step
            optimizer: (default = sgd) which optimizer to use to train the model
            with_lr_scheduler: (default = False) whether to use the learning rate scheduler during training
            batch_size: (default = 245) the training batch size
            temperature: (default = 0.5) the SimCLR temperature parameter
        """
        super(ContrastiveConsistentKNextStateRepresentationEnsembleTrainer, self).__init__(
            ensemble,
            learning_rate,
            optimizer,
            with_lr_scheduler=with_lr_scheduler)
        self._loss = SimCLR_Loss(batch_size, temperature)

    def _train_step_variable_length(self, batch: EnvironmentContrastiveBatch) -> float:
        """
        Update the model on the given batch of data
        Args:
            batch: samples with which to update the model

        Returns:
            average batch loss
            batch accuracy
        """
        # need to zero out the gradients
        self._optim.zero_grad()

        # do the forward pass of predictions
        predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

        loss = 0.
        # compute the loss for each model in the ensemble
        for (predicted_next_state_embed, next_state_embed) in zip(predicted_next_state_embeds, next_state_embeds):
            # check if all sequences in the batch had the same number of steps
            if batch.states.batch_sizes[0] == batch.states.batch_sizes[-1]:
                # compute cosine similarity on the batch of data and negate
                batch_loss = self._loss(predicted_next_state_embed[-1], next_state_embed[-1])
            else:
                # get the batch size for each predicted step
                batch_sizes = batch.states.batch_sizes
                sequence_end_indxs = torch.where(batch_sizes[:-1] > batch_sizes[1:])[0]
                # compute the length of each sequence
                sequence_end_indxs_count = list(zip(sequence_end_indxs.tolist(),
                                                    (batch_sizes[sequence_end_indxs] - batch_sizes[sequence_end_indxs + 1]).tolist()))
                # check if the lengths of all sequences in the batch are accounted for
                if len(sequence_end_indxs_count) < batch_sizes[0]:
                    # any sequences in the batch not accounted for have the complete length
                    sequence_end_indxs_count.append((batch_sizes.shape[0] - 1,
                                                     batch_sizes[0].item() - len(sequence_end_indxs_count)))
                # pull out the k-th prediction and true latent next state representation
                kth_predicted_next_state_embed, kth_next_state_embed = [], []
                for final_indx, count in sequence_end_indxs_count:
                    kth_predicted_next_state_embed.append(predicted_next_state_embed[final_indx][-count:])
                    kth_next_state_embed.append(next_state_embed[final_indx][-count:])
                # compute cosine similarity on the batch of data and negate
                batch_loss = self._loss(torch.vstack(kth_predicted_next_state_embed),
                                        torch.vstack(kth_next_state_embed))

            loss += torch.mean(batch_loss)
        # compute the gradients
        loss.backward()
        # apply the gradients to model parameters
        self._optim.step()

        return loss.item() / len(self._ensemble)

    def _train_step(self, batch: t.List[EnvironmentContrastiveBatch]) -> t.Tuple[float, float]:
        """
        Update the model on the given batch of data
        Args:
            batch: samples with which to update the model

        Returns:
            average batch loss
            batch accuracy
        """
        # need to zero out the gradients
        self._optim.zero_grad()

        # do the forward pass of predictions
        predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

        loss = 0.
        accuracy = 0.
        # compute the loss for each model in the ensemble
        for (predicted_next_state_embed, next_state_embed) in zip(predicted_next_state_embeds, next_state_embeds):
            # get the batch size and the number of time steps
            batch_size, time_steps, embed_size = predicted_next_state_embed.size()
            # compute the SimCLR loss on the batch of data and accumulate
            batch_loss, batch_accuracy = self._loss(predicted_next_state_embed.view((batch_size*time_steps, embed_size)),
                                                    next_state_embed.view((batch_size*time_steps, embed_size)))
            loss += batch_loss
            accuracy += batch_accuracy
        # compute the gradients
        loss.backward()
        # apply the gradients to model parameters
        self._optim.step()

        return loss.item() / len(self._ensemble), accuracy / len(self._ensemble)

    def evaluate(
            self,
            test_loader: DataLoader) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model on the given data
        Args:
            test_loader: data loader for data the model will be evaluated on
                         much generate batches from a EnvironmentTransitionDataset

        Returns:
            mean loss on the evaluation set
        """
        # set the model in evaluation mode to avoid gradients
        self._ensemble.eval()

        # accumulate the evaluation losses and accuracies
        losses = []
        accuracies = []

        with torch.no_grad():
            # evaluate the model on each batch of data
            for batch in test_loader:
                # do the forward pass of predictions
                # compute the predicted preference labels
                predicted_next_state_embeds, next_state_embeds = self._shared_step(batch)

                # we need to initialize the loss to 0 to accumulate across networks
                loss = 0.
                accuracy = 0.
                # compute the loss for each model in the ensemble
                for (predicted_next_state_embed, next_state_embed) in zip(predicted_next_state_embeds, next_state_embeds):
                    # get the batch size and the number of time steps
                    batch_size, time_steps, embed_size = predicted_next_state_embed.size()
                    # compute the SimCLR loss on the batch of data and accumulate
                    batch_loss, batch_accuracy = self._loss(
                        predicted_next_state_embed.view((batch_size * time_steps, embed_size)),
                        next_state_embed.view((batch_size * time_steps, embed_size)))
                    loss += batch_loss
                    accuracy += batch_accuracy
                # track the mean error
                losses.append(loss.item() / len(self._ensemble))
                accuracies.append(accuracy / len(self._ensemble))

        return np.asarray(losses), np.asarray(accuracies)
