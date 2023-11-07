#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import typing as t
from pathlib import Path
import time
import shutil

import yaml

from zipfile import ZipFile

import numpy as np

import torch
import torch.functional as F

from BPref.replay_buffer import TrajectoryReplayBuffer

from reed.data.preprocess_images import PreProcessInference


PREFERENCE_TRIPLET = t.Tuple[np.ndarray, np.ndarray, np.ndarray]
PREFERENCE_TRIPLET_BATCH = t.Tuple[np.ndarray, np.ndarray, np.ndarray]


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
        current_index = current_index[0:max_index] + current_index[max_index + 1:]

        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs,
            obs[selected_index]],
            axis=0)
    return selected_index


def compute_smallest_dist(obs, full_obs, device: torch.device):
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
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device),
                            dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)

        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class _PreferenceLabeller:
    def __init__(self, label_margin: float = 0.0, teacher_beta: float = -1, teacher_gamma: float = 1,
                 teacher_eps_mistake: float = 0, teacher_eps_skip: float = 0, teacher_eps_equal: float = 0):
        """
        Assigns preference labels to the trajectory pairs following the strategy specified by the parameters

        Args:
            label_margin:
            teacher_beta
            teacher_gamma: used to determine how much influence each reward has on the preference label based on
                           order within the trajectory. Used to compute the return
            teacher_eps_mistake: the frequency with which the teacher assigns an incorrect label
            teacher_eps_skip: the frequency with which the teacher does not assign a label
            teacher_eps_equal: the maximum difference between trajectory returns for the two trajectories to be labelled
                               as equally preferred
        """
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        self.label_margin = label_margin
        self.label_target = 1 - 2 * self.label_margin

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        """
        For each trajectory pair, assign a preference label

        Assigning a preference label can involve not labelling a trajectory pair, in which case the trajectory pair
        is removed from trajectories one and trajectories two

        Args:
            sa_t_1: the state-action pairs from trajectories one
            sa_t_2: the state-action pairs from trajectories two
            r_t_1: the reward per transition in the trajectories one
            r_t_2: the reward per transition in the trajectories two
        """
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
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], dim=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1).to_numpy()
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


class PreferenceDataset:
    def __init__(self, observation_dim: t.Union[t.Tuple, int], action_dim: t.Union[t.Tuple, int], capacity: int,
                 size_segment: int, out_path: Path, image_observations: bool, grayscale_images: bool,
                 collect_image_pref_dataset: bool, state_action_formatter: PreProcessInference,
                 teacher_beta: float = -1, teacher_gamma: float = 1,
                 teacher_eps_mistake: float = 0, teacher_eps_skip: float = 0, teacher_eps_equal: float = 0):
        """
        Args:
            observation_dim: the dimensionality of the observations
            action_dim: the dimensionality of the actions
            capacity: the maximum number of trajectory pairs to include in the action_dimtaset
            size_segment: the length of the trajectory segments
            out_path: the location where the preference action_dimtaset will be written to disk during training
            image_observations: whether the observations given to the reward model are images
            grayscale_images: whether the image observations should be converted to grayscale instead of color
            collect_image_pref_dataset: whether to collect the image preference dataset separate from the observations.
                                        Should NOT be set to true if the observations are images.
            state_action_formatter: function that maps states and actions to a single input
            teacher_beta
            teacher_gamma: used to determine how much influence each reward has on the preference label based on
                           order within the trajectory. Used to compute the return
            teacher_eps_mistake: the frequency with which the teacher assigns an incorrect label
            teacher_eps_skip: the frequency with which the teacher does not assign a label
            teacher_eps_equal: the maximum difference between trajectory returns for the two trajectories to be labelled
                               as equally preferred
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.size_segment = size_segment
        self.out_path = out_path
        self.image_observations = image_observations
        self.grayscale_images = grayscale_images
        # whether to collect the preference dataset as images
        # only needs to be set to True if we are not learning the reward function from images
        # if we are learning the reward function from images then we have an image dataset
        self.collect_image_pref_dataset = collect_image_pref_dataset

        # formats the state-action pairs into a single input to the reward model
        self.state_action_formatter = state_action_formatter

        # track where each preference triplet is written to disk
        self._preference_triplet_tracker: t.List[Path] = []

        self.buffer_index = 0
        self.buffer_full = False

        # create the preference labeller
        self._preference_labeller = _PreferenceLabeller(teacher_beta=teacher_beta, teacher_gamma=teacher_gamma,
                                                        teacher_eps_mistake=teacher_eps_mistake,
                                                        teacher_eps_skip=teacher_eps_skip,
                                                        teacher_eps_equal=teacher_eps_equal)

        # make sure the outpath where the trajectories will be written exist
        self.out_path.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self._preference_triplet_tracker)

    def __getitem__(self, item: int) -> PREFERENCE_TRIPLET:
        """
        Load and return the preference triplet at the specified index in the buffer

        Args:
            item: index of the triplet in the buffer
        Returns:
            trajectory one
            trajectory two
            preference label
        """
        # get the location of the specified preference triplet and load it into memory
        npz_archive = np.load(self._preference_triplet_tracker[item].as_posix())

        # grab the trajectories and preference labels
        trajectory_one = npz_archive["trajectory_one"]
        trajectory_two = npz_archive["trajectory_two"]
        preference_label = npz_archive["preference_label"]

        return trajectory_one, trajectory_two, preference_label

    def get_batch(self, indices: t.List[int]) -> PREFERENCE_TRIPLET_BATCH:
        """
        Load and return the batch of preference triplets at the given indices in the buffer

        Args:
             indices: the buffer indices of the preference triplets to load into memory
        Returns:
            batch of trajectories one
            batch of trajectories two
            batch of preference labels
        """
        # accumulate the trajectory pairs and preference labels
        trajectories_one = []
        trajectories_two = []
        preference_labels = []
        # grab each preference triplet
        for index in indices:
            trajectory_one, trajectory_two, preference_label = self[index]
            trajectories_one.append(np.expand_dims(trajectory_one, axis=0))
            trajectories_two.append(np.expand_dims(trajectory_two, axis=0))
            preference_labels.append(preference_label)

        return (np.concatenate(trajectories_one, axis=0), np.concatenate(trajectories_two, axis=0),
                np.concatenate(preference_labels, axis=0))

    def _sample_trajectory_segments_uniform(self,
                                            experience_buffer: TrajectoryReplayBuffer,
                                            trajectory_count: int,
                                            mini_batch_size: int) -> t.Tuple[np.ndarray, np.ndarray, t.Optional[np.ndarray]]:
        """
        Uniformly sample trajectories and then uniformly sample a segment of the trajectory.

        Format and track the state-action pairs from each trajectory segment
        Format and track rewards from each trajectory segment

        Combine the formatted state-action pairs and the rewards across trajectory segments

        Args:
            experience_buffer: the replay buffer from which trajectory pairs will be drawn
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
            trajectory = experience_buffer.trajectories[traj_index]
            # select a random segment from the trajectory
            traj_segment = trajectory.random_segment(length=self.size_segment)
            # track the rewards associated with the random segment
            rewards.append(np.expand_dims(traj_segment.env_rewards, axis=0))
            # format the state and action based on whether image observations are being used
            if self.image_observations:
                formatted_pair = self.state_action_formatter.format_state_action(
                    traj_segment.initial_image_observations,
                    traj_segment.actions,
                    batch_sa=True)
            else:
                formatted_pair = self.state_action_formatter.format_state_action(
                    traj_segment.initial_observations,
                    traj_segment.actions,
                    batch_sa=True)
                if self.collect_image_pref_dataset:
                    image_observations.append(np.expand_dims(traj_segment.initial_image_observations, axis=0))
            # add a dimension in the front so we can concatenate later and the track
            state_action_pairs.append(np.expand_dims(formatted_pair, axis=0))
        return (np.concatenate(state_action_pairs, axis=0),
                np.concatenate(rewards, axis=0),
                (np.concatenate(image_observations, axis=0) if image_observations is not None else None))

    @staticmethod
    def get_rank_probability(trajectories_one: np.ndarray, trajectories_two: np.ndarray,
                             reward_model: torch.nn.Module):
        """
        Compute the preference-prediction disagreement between the ensemble members for each trajectory pair

        Args:
            trajectories_one: the trajectories one to be evaluated for ensemble disagreement
            trajectories_two: the trajectories two to be evaluated for ensemble disagreement
            reward_model: the ensemble of networks that will be used to compute disagreement
        """

        # get probability x_1 > x_2
        probs = []
        for member in range(len(reward_model.ensemble)):
            probs.append(reward_model.p_hat_member(trajectories_one,
                                                   trajectories_two,
                                                   member=member).cpu().numpy())
        probs = np.array(probs)

        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_queries(self, experience_buffer: TrajectoryReplayBuffer, mb_size=20):
        len_traj, max_len = experience_buffer.trajectory_lengths[0], experience_buffer.trajectory_count

        # if len(self.experience_buffer.trajectory_lengths[0][-1]) < len_traj:
        # check that the last trajectory contains at least as many transitions as the target segment length
        # we check the last trajectory, because it may be incomplete
        # this is a carry over from the original code. The authors had an assumption that all "completed" trajectories
        # will be at least as long as the target segment length
        if experience_buffer.trajectory_lengths[-1] < self.size_segment:
            max_len = max_len - 1

        # grab each trajectory, select a random segment from each, format the state-action pairs, and concatenate
        # along the batch dimension
        state_action_pair_traj_one, r_t_1, images_traj_one = self._sample_trajectory_segments_uniform(
            experience_buffer=experience_buffer,
            trajectory_count=max_len,
            mini_batch_size=mb_size)
        state_action_pair_traj_two, r_t_2, images_traj_two = self._sample_trajectory_segments_uniform(
            experience_buffer=experience_buffer,
            trajectory_count=max_len,
            mini_batch_size=mb_size)
        # confirm the image-specific variables are only populated when they should be
        if not self.collect_image_pref_dataset and self.image_observations:
            assert images_traj_one is None and images_traj_two is None
        return state_action_pair_traj_one, state_action_pair_traj_two, r_t_1, r_t_2, images_traj_one, images_traj_two

    def put_queries(self, state_action_pair_traj_one: np.ndarray, state_action_pair_traj_two: np.ndarray,
                    preference_labels: np.ndarray,
                    images_traj_one: t.Optional[np.ndarray] = None, images_traj_two: t.Optional[np.ndarray] = None):
        """
        Args:
            state_action_pair_traj_one: the state-action pairs that make up the trajectories one in the queries
            state_action_pair_traj_two: the state-action pairs that make up the trajectories two in the queries
            preference_labels: the preference labels for each pair of trajectories
            images_traj_one: the images for trajectories one
            images_traj_two: the images for trajectories two
        """
        # get the number of triplets to be stored
        total_sample = state_action_pair_traj_one.shape[0]
        # write each preference_triplet to disk
        for batch_indx in range(total_sample):
            # get the index of the triplet in the "buffer"
            preference_triplet_index = self.buffer_index + batch_indx
            # check if we need to wrap the buffer
            if preference_triplet_index >= self.capacity:
                preference_triplet_index -= self.capacity
            elif not self.buffer_full:
                # this is a previously unseen preference triplet buffer index, so we need to track the triplet location
                self._preference_triplet_tracker.append(self.out_path / f"preference_triplet_{preference_triplet_index}.npz")
            # save the preference triplet
            np.savez((self.out_path / f"preference_triplet_{preference_triplet_index}.npz").as_posix(),
                     trajectory_one=state_action_pair_traj_one[batch_indx],
                     trajectory_two=state_action_pair_traj_two[batch_indx],
                     preference_label=preference_labels[batch_indx],
                     image_trajectory_one=(
                         None if images_traj_one is None else images_traj_one[batch_indx]),
                     image_trajectory_two=(
                         None if images_traj_two is None else images_traj_two[batch_indx]))
        # set the new buffer index
        next_index = self.buffer_index + total_sample
        # check if the buffer has wrapped
        if next_index >= self.capacity:
            self.buffer_full = True
            # wrap the buffer index
            self.buffer_index = next_index - self.capacity
        else:
            self.buffer_index = next_index

    def uniform_sampling(self, experience_buffer: TrajectoryReplayBuffer, mb_size: int) -> int:
        """
        Grow the preference dataset with preference triplets uniformly sampled from the experience buffer

        Args:
            experience_buffer: the replay buffer from which to sample trajectory pairs
            mb_size: target number of preference triplets to add to the preference dataset. Fewer than the target may
                     be added depending on the whether labeller skips labelling some trajectories.
        Returns:
            number of preference triplets added to the dataset
        """
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, img_sa_t_1, img_sa_t_2 = self.get_queries(experience_buffer=experience_buffer,
                                                                                mb_size=mb_size)

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self._preference_labeller.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, img_sa_t_1, img_sa_t_2)

        return len(labels)

    # TODO: refactor to break the circular import that would need to happen in order to specify that reward_model here
    #  should be BPref.reward_model.RewardModel
    def disagreement_sampling(self, experience_buffer: TrajectoryReplayBuffer, mb_size: int, large_batch: int,
                              reward_model: torch.nn.Module) -> int:
        """
        Grow the preference dataset with preference triplets from the experience buffer that the reward ensemble
        disagrees about

        Args:
            experience_buffer: the replay buffer from which to sample trajectory pairs
            mb_size: target number of preference triplets to add to the preference dataset. Fewer than the target may
                     be added depending on the whether labeller skips labelling some trajectories.
            large_batch: scales up the number of triplets to add to the preference dataset to uniformly select a large
                         number of trajectory pairs, which are then pruned based on which ones the reward ensemble
                         has the most disagreement over
            reward_model: the ensemble of reward networks that will be used to assess disagreement.
                          Should be BPref.reward_model.RewardModel, but cannot import and reference from here right now
                          as it would lead to circular imports
        Returns:
            number of preference triplets added to the dataset
        """
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, img_sa_t_1, img_sa_t_2 = self.get_queries(
            experience_buffer=experience_buffer, mb_size=mb_size * large_batch)

        # get final queries based on ensemble member disagreement
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2, reward_model=reward_model)
        top_k_index = (-disagree).argsort()[:mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        if img_sa_t_1 is not None:
            img_sa_t_1 = img_sa_t_1[top_k_index]
            img_sa_t_2 = img_sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self._preference_labeller.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, img_sa_t_1, img_sa_t_2)

        return len(labels)

    def set_teacher_thres_skip(self, new_margin):
        self._preference_labeller.teacher_thres_skip = new_margin * self._preference_labeller.teacher_eps_skip

    def set_teacher_thres_equal(self, new_margin):
        self._preference_labeller.teacher_eps_equal = new_margin * self._preference_labeller.teacher_eps_equal

    def save(self, dataset_dir: Path, env_id: str, step: int):
        """
        Saves the preference dataset as a zip archive and the labeller configuration as a yaml to the specified location

        Args:
            dataset_dir: path where the dataset is to be saved
            env_id: the environment/task within which the data was generated
            step: the number of policy training steps taken to produce this dataset
        """
        # create the ZipFile object
        zip_obj = ZipFile(dataset_dir / f"{env_id}_preference_dataset_{step}.zip", "w")
        # the configuration for the online preference dataset
        config = {"teacher_params": {"teacher_beta": self._preference_labeller.teacher_beta,
                                     "teacher_gamma": self._preference_labeller.teacher_gamma,
                                     "teacher_eps_mistake": self._preference_labeller.teacher_eps_mistake,
                                     "teacher_eps_equal": self._preference_labeller.teacher_eps_equal,
                                     "teacher_eps_skip": self._preference_labeller.teacher_eps_skip,
                                     "teacher_thres_skip": self._preference_labeller.teacher_thres_skip,
                                     "teacher_thres_equal": self._preference_labeller.teacher_thres_equal,
                                     "label_margin": self._preference_labeller.label_margin,
                                     "label_target": self._preference_labeller.label_target}}
        with open((dataset_dir / f"preference_dataset_config.yaml").as_posix(), "w+") as f:
            yaml.dump(config, f)
        # write the labeller config to the preference dataset's zip archive
        zip_obj.write(dataset_dir / f"preference_dataset_config.yaml")

        # add each preference triplet to the zip archive
        for pref_triplet_path in self._preference_triplet_tracker:
            zip_obj.write(pref_triplet_path)
            # move the  file from it temp location to the artifact directory
            file_dest_path = dataset_dir / pref_triplet_path.name
            shutil.move(pref_triplet_path, file_dest_path)
        # close the Zip File
        zip_obj.close()
