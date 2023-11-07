#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import OrderedDict

import yaml
import typing as t
from pathlib import Path

from pebble_self_future_consistency.utils.load_from_blobby import load_pretrained_model
from BPref.replay_buffer import TrajectoryReplayBuffer
from pebble_self_future_consistency.preference_dataset import OnlinePreferenceDataset

from pebble_self_future_consistency.reward_model import StateActionNetwork, ImageStateActionNetwork


def gen_img_net(in_size: t.List[int], out_size: int,
                H=128, n_layers=3, activation='tanh',
                image_encoder_architecture="pixl2r"):
    return ImageStateActionNetwork(in_size, out_size,
                                   hidden_dim=H, hidden_depth=n_layers,
                                   final_activation=activation,
                                   image_encoder_architecture=image_encoder_architecture)


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    return StateActionNetwork(in_size,
                              out_size=out_size,
                              hidden_dim=H,
                              hidden_depth=n_layers,
                              final_activation=activation).float()

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index


def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class RewardModel:
    def __init__(self, ds, da, experience_buffer: TrajectoryReplayBuffer,
                 ensemble_size=3, lr=3e-4, mb_size=128, reward_train_batch=128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 normalize_states: bool = False,
                 image_observations: bool = False,
                 image_encoder_architecture: str = "pixl2r",
                 image_hidden_num_channels: int = 32,
                 grayscale_images: bool = True,
                 collect_image_pref_dataset: bool = False,
                 preference_dataset_outpath: str = "/tmp/preference_dataset/",
                 device: torch.device = "cuda",
                 multi_gpu: bool = False):
        # the device the model will be put on
        self.device = device
        # whether data parallelism should be used during model training
        self.multi_gpu = multi_gpu
        # train data is trajectories, must process to sa and s..
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.optimizer = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        # track whether we are using image observations
        self.image_observations = image_observations
        # which image encoder to use
        self.image_encoder_architecture = image_encoder_architecture
        # number of channels to us in the image encoder's hidden layers
        self._image_hidden_num_channels = image_hidden_num_channels
        # whether to collect the preference dataset as images
        # only needs to be set to True if we are not learning the reward function from images
        # if we are learning the reward function from images then we have an image dataset
        self.collect_image_pref_dataset = collect_image_pref_dataset
        # whether to convert the images to grayscale
        self.grayscale_images = grayscale_images
        # the location where the preference dataset will be written to disk
        self.preference_dataset_outpath = preference_dataset_outpath

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        self.label_margin = label_margin
        self.label_target = 1 - 2 * self.label_margin

        # initialize the preference dataset that contains the synthetic preference labeller and will be used to train
        # the reward model
        self._initialize_preference_dataset()

        # compute the dimensions of the input to the reward function
        if not self.image_observations:
            self.reward_in_dim: int = self.ds+self.da
        else:
            # we need to concatenate the actions to last dimension of the image
            # the input to the reward net also needs to have the channels first
            # the image dimensions are given to us a (height, width, channels)
            sample_shape = list(self.ds)
            if self.grayscale_images:
                num_channels = self.da + 1
            else:
                num_channels = sample_shape[-1] + self.da
            # the dimensions of the input to the reward model
            self.reward_in_dim: t.List[int] = [num_channels] + sample_shape[:-1]
        # construct the reward ensemble
        self.construct_ensemble()

        # track the replay buffer from which we will sample trajectory pairs to be sent to the labeller
        self.experience_buffer = experience_buffer

        # parameters used to train the reward model on the preference labelled trajectories
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        # self.train_batch_size = 128
        self.train_batch_size = reward_train_batch
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch

        # whether the states should be mean and standard deviation normalized
        self._normalize_states = normalize_states

    def _initialize_preference_dataset(self):
        self.preference_dataset = OnlinePreferenceDataset(observation_dim=self.ds, action_dim=self.da, capacity=self.capacity,
                                                          size_segment=self.size_segment,
                                                          out_path=Path(self.preference_dataset_outpath),
                                                          image_observations=self.image_observations,
                                                          grayscale_images=self.grayscale_images,
                                                          collect_image_pref_dataset=self.collect_image_pref_dataset,
                                                          state_action_formatter=self.format_state_action,
                                                          teacher_beta=self.teacher_beta, teacher_gamma=self.teacher_gamma,
                                                          teacher_eps_mistake=self.teacher_eps_mistake,
                                                          teacher_eps_skip=self.teacher_eps_skip,
                                                          teacher_eps_equal=self.teacher_eps_equal
                                                          )

    def _sample_trajectory_segments_uniform(self,
                                            trajectory_count: int,
                                            mini_batch_size: int) -> t.Tuple[np.ndarray, np.ndarray, t.Optional[np.ndarray]]:
        """
        Uniformly sample trajectories and then uniformly sample a segment of the trajectory.

        Format and track the state-action pairs from each trajectory segment
        Format and track rewards from each trajectory segment

        Combine the formatted state-action pairs and the rewards across trajectory segments

        Args:
            trajectory_count: the number of trajectories to be sampled from
            mini_batch_size: the number of trajectories to sample

        Returns:
            the formatted state-action pairs from random trajectory segments from trajectories
            the rewards from each random trajectory segment
            (optionally) the image observations from each random trajectory segment - only returned when the flag to
                         collect image observations in the preference dataset is true and image observations are not
                         used to train the reward model
        """
        # select the trajectories to be included in this batch of trajectory segments
        trajectory_indices = np.random.choice(trajectory_count, size=mini_batch_size, replace=True)

        # accumulate the formatted state-action pairs and rewards from each trajectory segment
        state_action_pairs = []
        rewards = []
        # optionally accumulate image observations
        image_observations = ([] if self.collect_image_pref_dataset and not self.image_observations else None)
        # extract each trajectory and randomly sample a segment
        for traj_index in trajectory_indices:
            # grab the trajectory
            trajectory = self.experience_buffer.trajectories[traj_index]
            # select a random segment from the trajectory
            traj_segment = trajectory.random_segment(length=self.size_segment)
            # track the rewards associated with the random segment
            rewards.append(np.expand_dims(trajectory.rewards, axis=0))
            # format the state and action based on whether image observations are being used
            if self.image_observations:
                formatted_pair = self.format_state_action(traj_segment.initial_image_observations,
                                                          traj_segment.actions,
                                                          batch_sa=True)
            else:
                formatted_pair = self.format_state_action(traj_segment.initial_observations,
                                                          traj_segment.actions,
                                                          batch_sa=True)
                if self.collect_image_pref_dataset:
                    image_observations.append(np.expand_dims(trajectory.initial_image_observations, axis=0))
            # add a dimension in the front so we can concatenate later and the track
            state_action_pairs.append(np.expand_dims(formatted_pair, axis=0))
        return (np.concatenate(state_action_pairs, axis=0),
                np.concatenate(rewards, axis=0),
                (np.concatenate(image_observations, axis=0) if image_observations is not None else None))

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    def construct_ensemble(self):
        for _ in range(self.de):
            if self.image_observations:
                model = ImageStateActionNetwork(self.reward_in_dim,
                                                out_size=1,
                                                hidden_dim=256,
                                                hidden_depth=3,
                                                final_activation=self.activation,
                                                image_encoder_architecture=self.image_encoder_architecture,
                                                image_hidden_num_channels=self._image_hidden_num_channels).float()
            else:
                model = StateActionNetwork(self.reward_in_dim,
                                           out_size=1,
                                           hidden_dim=256,
                                           hidden_depth=3,
                                           final_activation=self.activation).float()
            print(model)
            # track all model parameters
            self.paramlst.extend(model.parameters())

            # check if the model will be run with Data Parallelism
            if self.multi_gpu:
                print(f"There are {torch.cuda.device_count()} GPU devices, so the reward ensemble WILL be trained "
                      f"using nn.DataParallel")
                self.ensemble.append(nn.DataParallel(model).to(self.device))
            else:
                print(f"There are {torch.cuda.device_count()} GPU devices, so the reward ensemble will NOT be trained "
                      f"using nn.DataParallel")
                self.ensemble.append(model.to(self.device))

        # create a single optimizer applied to all ensemble members
        self.optimizer = torch.optim.Adam(self.paramlst, lr=self.lr)

    def format_state(self, obs, batch_states: bool = False):
        if self.image_observations:
            if batch_states:
                # check if the images needs to be converted to grayscale
                if self.grayscale_images:
                    obs = obs.astype(float)
                    obs[:, :, :, 0] *= 0.1140
                    obs[:, :, :, 1] *= 0.587
                    obs[:, :, :, 2] *= 0.2989
                    obs = np.sum(obs, axis=-1, keepdims=True)
                # permute the input so that the channels are in the first dimension
                obs = np.transpose(obs, (0, 3, 1, 2))
                return obs
            else:
                # check if the images needs to be converted to grayscale
                if self.grayscale_images:
                    obs = obs.astype(float)
                    obs[:, :, 0] *= 0.1140
                    obs[:, :, 1] *= 0.587
                    obs[:, :, 2] *= 0.2989
                    obs = np.sum(obs, axis=-1, keepdims=True)
                # permute the input so that the channels are in the first dimension
                obs = np.transpose(obs, (2, 0, 1))
                # add a dimension along the front for concatenation into the buffer
                return obs.reshape(1, *obs.shape)
        else:
            return obs.reshape(1, self.ds)

    def format_state_action(self, obs, act, batch_sa: bool = False, by_trajectory: bool = False) -> np.ndarray:
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
            if batch_sa:
                # check if the images needs to be converted to grayscale
                if self.grayscale_images:
                    obs = obs.astype(float)
                    obs[:, :, :, 0] *= 0.1140
                    obs[:, :, :, 1] *= 0.587
                    obs[:, :, :, 2] *= 0.2989
                    obs = np.sum(obs, axis=-1, keepdims=True)
                # we concatenate the actions along channel dimension of the image
                if by_trajectory:
                    repeated_actions = np.tile(act.reshape(act.shape[0], act.shape[1], 1, 1, act.shape[-1]),
                                               (1, 1, self.ds[0], self.ds[1], 1))
                else:
                    repeated_actions = np.tile(act.reshape(act.shape[0], 1, 1, act.shape[-1]),
                                               (1, self.ds[0], self.ds[1], 1))
                # now concatenate the two
                sa_t = np.concatenate((obs, repeated_actions), axis=-1)
                # permute the input so that the channels are in the first dimension
                if by_trajectory:
                    sa_t = np.transpose(sa_t, (0, 1, 4, 2, 3))
                else:
                    sa_t = np.transpose(sa_t, (0, 3, 1, 2))
                return sa_t
            else:
                # check if the images needs to be converted to grayscale
                if self.grayscale_images:
                    obs = obs.astype(float)
                    obs[:, :, 0] *= 0.1140
                    obs[:, :, 1] *= 0.587
                    obs[:, :, 2] *= 0.2989
                    obs = np.sum(obs, axis=-1, keepdims=True)
                # we concatenate the actions along channel dimension of the image
                repeated_actions = np.tile(act.reshape(1, 1, -1), (self.ds[0], self.ds[1], 1))
                # now concatenate the two
                sa_t = np.concatenate((obs, repeated_actions), axis=-1)
                # permute the input so that the channels are in the first dimension
                sa_t = np.transpose(sa_t, (2, 0, 1))
                # add a dimension along the front for concatenation into the buffer
                return sa_t.reshape(1, *self.reward_in_dim)
        else:
            sa_t = np.concatenate([obs, act], axis=-1)
            if batch_sa:
                return sa_t
            else:
                return sa_t.reshape(1, self.da + self.ds)
        
    def get_rank_probability(self, x_1: np.ndarray, x_2: np.ndarray):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

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
                        mb_rhat1 = self.r_hat_member(x_1[start_indx:start_indx + mb_size].reshape((-1, *x_1.shape[2:])),
                                                     member=member).detach().cpu().reshape((mb_size, x_1.shape[1], 1))
                        mb_rhat2 = self.r_hat_member(x_2[start_indx:start_indx + mb_size].reshape((-1, *x_2.shape[2:])),
                                                     member=member).detach().cpu().reshape((mb_size, x_2.shape[1], 1))
                    else:
                        remainder_mb_size = x_1.shape[0] - start_indx
                        # process the leftover trajectories in a batch smaller than mb_size
                        mb_rhat1 = self.r_hat_member(x_1[start_indx:].reshape((-1, *x_1.shape[2:])),
                                                     member=member).detach().cpu()
                        mb_rhat1 = mb_rhat1.reshape((remainder_mb_size, x_1.shape[1], 1))
                        mb_rhat2 = self.r_hat_member(x_2[start_indx:].reshape((-1, *x_2.shape[2:])),
                                                     member=member).detach().cpu()
                        mb_rhat2 = mb_rhat2.reshape((remainder_mb_size, x_2.shape[1], 1))
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

    def r_hat_member(self, x: np.ndarray, member: int = -1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def r_hat(self, x: np.ndarray):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the
        # rewards are already normalized and I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, env_id, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), f'{model_dir}/{env_id}_reward_model_{step}_{member}.pt'
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )

    def from_pretrained(self,
                        blobby_uris_filepath: t.Optional[Path] = None,
                        blobby_artifact_uri: t.Optional[Path] = None):
        """
        Load the ensemble parameters from the pretrained networks.

        Args:
            blobby_uris_filepath: (optional) the location of the YAML file with the Blobby URIs where
                                  the pretrained network parameters are stored on Blobby
            blobby_artifact_uri: (optional) blobby uri for the Bolt task with the pre-trained reward model artifacts
        """
        if blobby_uris_filepath is not None:
            assert blobby_uris_filepath.exists()
            with open(blobby_uris_filepath, "r") as f:
                blobby_uris = yaml.load(f, Loader=yaml.FullLoader)
            assert len(blobby_uris) == len(self.ensemble), (f"The number of networks in the reward ensemble "
                                                            f"({len(self.ensemble)}) and the number of "
                                                            f"Blobby URIs ({len(blobby_uris)}) much match.")

            for indx, blobby_uri in enumerate(blobby_uris):
                state_dict = load_pretrained_model(blobby_uri)
                self.ensemble[indx].load_state_dict(state_dict)
                self.ensemble[indx].to(self.device)
        elif blobby_artifact_uri is not None:
            for indx in range(len(self.ensemble)):
                state_dict = load_pretrained_model(f"{blobby_artifact_uri}_{indx}.pt")
                self.ensemble[indx].load_state_dict(state_dict)
                self.ensemble[indx].to(self.device)
        else:
            raise NotImplementedError

    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20):
        len_traj, max_len = self.experience_buffer.trajectory_lengths[0], self.experience_buffer.trajectory_count
        
        # if len(self.experience_buffer.trajectory_lengths[0][-1]) < len_traj:
        # check that the last trajectory contains at least as many transitions as the target segment length
        # we check the last trajectory, because it may be incomplete
        # this is a carry over from the original code. The authors had an assumption that all "completed" trajectories
        # will be at least as long as the target segment length
        if self.experience_buffer.trajectory_lengths[-1] < self.size_segment:
            max_len = max_len - 1

        # grab each trajectory, select a random segment from each, format the state-action pairs, and concatenate
        # along the batch dimension
        sa_t_1, r_t_1, img_sa_t_1 = self._sample_trajectory_segments_uniform(trajectory_count=max_len,
                                                                             mini_batch_size=mb_size)
        sa_t_2, r_t_2, img_sa_t_2 = self._sample_trajectory_segments_uniform(trajectory_count=max_len,
                                                                             mini_batch_size=mb_size)
        # confirm the image-specific variables are only populated when they should be
        if not self.collect_image_pref_dataset and self.image_observations:
            assert img_sa_t_1 is None and img_sa_t_2 is None

        return sa_t_1, sa_t_2, r_t_1, r_t_2, img_sa_t_1, img_sa_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels,
                    img_sa_t_1: t.Optional[np.ndarray] = None, img_sa_t_2: t.Optional[np.ndarray] = None):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if img_sa_t_1 is not None and self.img_buffer_seg1 is None:
            self.img_buffer_seg1 = np.empty((self.capacity, self.size_segment, *img_sa_t_1.shape[2:]), dtype=np.float32)
            self.img_buffer_seg2 = np.empty((self.capacity, self.size_segment, *img_sa_t_2.shape[2:]), dtype=np.float32)
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            if img_sa_t_1 is not None:
                np.copyto(self.img_buffer_seg1[self.buffer_index:self.capacity], img_sa_t_1[:maximum_index])
                np.copyto(self.img_buffer_seg2[self.buffer_index:self.capacity], img_sa_t_2[:maximum_index])

            remain = total_sample - maximum_index
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

                if img_sa_t_1 is not None:
                    np.copyto(self.img_buffer_seg1[0:remain], img_sa_t_1[maximum_index:])
                    np.copyto(self.img_buffer_seg2[0:remain], img_sa_t_2[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)

            if img_sa_t_1 is not None:
                np.copyto(self.img_buffer_seg1[self.buffer_index:next_index], img_sa_t_1)
                np.copyto(self.img_buffer_seg2[self.buffer_index:next_index], img_sa_t_2)
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    
    def kcenter_sampling(self):
        # TODO: refactor to use the new preference dataset class
        raise NotImplementedError("Not refactored to handle the new preference dataset class")
        # get queries
        num_init = self.mb_size*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),  
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_disagree_sampling(self):
        # TODO: refactor to use the new preference dataset class
        raise NotImplementedError("Not refactored to handle the new preference dataset class")
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        # TODO: refactor to use the new preference dataset class
        raise NotImplementedError("Not refactored to handle the new preference dataset class")
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size=num_init)
        
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def uniform_sampling(self) -> int:
        """
        Use uniform sampling to grow the preference dataset

        Returns:
            number of preference triplets added to the dataset
        """
        # TODO: refactor to fully move inside the preference dataset
        return self.preference_dataset.uniform_sampling(experience_buffer=self.experience_buffer,
                                                        mb_size=self.mb_size)
    
    def disagreement_sampling(self) -> int:
        """
        Use disagreement sampling to grow the preference dataset

        Returns:
            number of preference triplets added to the dataset
        """
        # TODO: refactor to fully move inside the preference dataset
        # TODO: refactor to break the circular import that would need to happen in order to specify that reward_model
        #  here should be BPref.reward_model.RewardModel
        return self.preference_dataset.disagreement_sampling(experience_buffer=self.experience_buffer,
                                                             mb_size=self.mb_size, large_batch=self.large_batch,
                                                             reward_model=self)
    
    def entropy_sampling(self):
        # TODO: refactor to use the new preference dataset class
        raise NotImplementedError("Not refactored to handle the new preference dataset class")
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = len(self.preference_dataset)
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1, sa_t_2, labels = self.preference_dataset.get_batch(idxs)
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                # check if we need to collapse the batch and time dimensions into one and then reconstruct the two
                if self.image_observations:
                    formatted_sa_t_1 = sa_t_1.reshape((-1, *sa_t_1.shape[2:]))
                    r_hat1 = self.r_hat_member(formatted_sa_t_1, member=member).reshape((sa_t_1.shape[0], sa_t_1.shape[1], 1))
                    formatted_sa_t_2 = sa_t_2.reshape((-1, *sa_t_2.shape[2:]))
                    r_hat2 = self.r_hat_member(formatted_sa_t_2, member=member).reshape((sa_t_2.shape[0], sa_t_2.shape[1], 1))
                else:
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)

                r_hat = torch.cat([r_hat1, r_hat2], dim=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

                # curr_loss.backward()
                # self.opt.step()
            loss.backward()
            self.optimizer.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = len(self.preference_dataset)

        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            loss = 0.0
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1, sa_t_2, labels = self.preference_dataset.get_batch(idxs)
                labels = torch.from_numpy(labels).to(self.device)

                if member == 0:
                    total += labels.size(0)
                
                # get logits
                # check if we need to collapse the batch and time dimensions into one and then reconstruct the two
                if self.image_observations:
                    formatted_sa_t_1 = sa_t_1.reshape((-1, *sa_t_1.shape[2:]))
                    r_hat1 = self.r_hat_member(formatted_sa_t_1, member=member).reshape(
                        (sa_t_1.shape[0], sa_t_1.shape[1], 1))
                    formatted_sa_t_2 = sa_t_2.reshape((-1, *sa_t_2.shape[2:]))
                    r_hat2 = self.r_hat_member(formatted_sa_t_2, member=member).reshape(
                        (sa_t_1.shape[0], sa_t_1.shape[1], 1))
                else:
                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat2 = self.r_hat_member(sa_t_2, member=member)

                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # check if the labels need to be converted from a single value (e.g. 0, 1) to multiple values with per
                # class (e.g. [1, 0])
                if labels.size()[-1] == 1:
                    labels = labels.flatten().long().to(self.device)
                    uniform_index = labels == -1
                    labels[uniform_index] = 0
                    targets = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                    targets += self.label_margin
                    if sum(uniform_index) > 0:
                        targets[uniform_index] = 0.5
                else:
                    targets = labels.float().to(self.device)
                # compute loss
                # print(F.softmax(r_hat, dim=-1))
                curr_loss = self.softXEnt_loss(r_hat, targets)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                # need to compute the maximum class for the labels
                _, max_labels = torch.max(labels.data, 1)
                correct = (predicted == max_labels).sum().item()
                ensemble_acc[member] += correct
            # print(loss)
            loss.backward()
            self.optimizer.step()

        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc

    def save_preference_dataset(self, dataset_dir: Path, env_id: str, step: int):
        """
        Save the accumulated preference dataset
        Args:
            dataset_dir: the directory where the accumulated preference dataset will be saved
            env_id: string identifier for the environment
            step: the number of policy training steps taken to produce this dataset
            
        * the dataset structure matches that af the OnlinePreferenceDataset in
          pebble_self_future_consistency.experiments.bootstrapping_reward_functions.preference_dataset
        """
        # save the teacher-labelled trajectory pairs
        np.savez((dataset_dir / f"trajectories_one_{step}.npz").as_posix(),
                 trajectories=self.buffer_seg1[:self.buffer_index])
        np.savez((dataset_dir / f"trajectories_two_{step}.npz").as_posix(),
                 trajectories=self.buffer_seg2[:self.buffer_index])
        np.savez((dataset_dir / f"trajectories_labels_{step}.npz").as_posix(),
                 labels=self.buffer_label[:self.buffer_index])

        if self.img_buffer_seg1 is not None:
            np.savez((dataset_dir / f"img_trajectories_one_{step}.npz").as_posix(),
                     trajectories=self.img_buffer_seg1[:self.buffer_index])
        if self.img_buffer_seg2 is not None:
            np.savez((dataset_dir / f"img_trajectories_two_{step}.npz").as_posix(),
                     trajectories=self.img_buffer_seg2[:self.buffer_index])

        # the configuration for the online preference dataset
        config = {"teacher_params": {"teacher_beta": self.teacher_beta,
                                     "teacher_gamma": self.teacher_gamma,
                                     "teacher_eps_mistake": self.teacher_eps_mistake,
                                     "teacher_eps_equal": self.teacher_eps_equal,
                                     "teacher_eps_skip": self.teacher_eps_skip,
                                     "teacher_thres_skip": self.teacher_thres_skip,
                                     "teacher_thres_equal": self.teacher_thres_equal,
                                     "label_margin": self.label_margin,
                                     "label_target": self.label_target}}
        with open((dataset_dir / f"preference_dataset_config.yaml").as_posix(), "w+") as f:
            yaml.dump(config, f)
