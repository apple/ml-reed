#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os, shutil
from pathlib import Path

import attr
import typing as t

from zipfile import ZipFile

import numpy as np
import torch

from reed.data.preprocess_images import PreProcessInference


@attr.s(frozen=True)
class Transition:
    """
    Stores multiple environment transitions and torch Tensors

    Transitions are distinct from trajectories, because there is no guaranteed temporal relationship
    between subsequent transitions
    """
    observation = attr.ib(type=np.ndarray)
    action = attr.ib(type=np.ndarray)
    next_observation = attr.ib(type=np.ndarray)
    reward = attr.ib(type=float)
    not_done = attr.ib(type=float)
    not_done_no_max = attr.ib(type=float)
    env_reward = attr.ib(type=t.Optional[float], default=None)
    image_observation = attr.ib(type=t.Optional[np.ndarray], default=None)
    next_image_observation = attr.ib(type=t.Optional[np.ndarray], default=None)

    def __attrs_post_init__(self):
        assert self.observation.shape == self.next_observation.shape


@attr.s(frozen=False)
class Trajectory:
    # TODO: Finish removing all old references to torch Tensors
    """
    Sequence of state transitions, actions, and rewards

    states: the states/observations from the course of the trajectory
    actions: the actions taken to trigger the transitions
    next_states: the states/observations post-actions
    rewards: the rewards received for the transitions
    """
    # the states, including initial state and end state
    _observations = attr.ib(type=np.ndarray)
    # the observations as images
    _image_observations = attr.ib(type=t.Optional[np.ndarray])
    # the actions taken to transition from one state to another
    actions = attr.ib(type=np.ndarray)
    # the rewards received for each state transition
    rewards = attr.ib(type=np.ndarray)
    # whether a transition resulted in the agent entering the
    # goal state
    not_dones = attr.ib(type=np.ndarray)
    # whether the goal state achieved or the episode timed out due to
    # reach the maximum allowed steps
    not_dones_no_max = attr.ib(type=np.ndarray)

    # the environment rewards associated with a transition
    env_rewards = attr.ib(type=t.Optional[np.ndarray], default=None)

    def __attrs_post_init__(self):
        if self._image_observations is None and self.env_rewards is None:
            assert (self._observations.shape[0] - 1 == self.actions.shape[0]
                    == self.rewards.shape[0] == self.not_dones.shape[0] == self.not_dones_no_max.shape[0])
        elif self._image_observations is None and self.env_rewards is not None:
            assert (self._observations.shape[0] - 1 == self.actions.shape[0]
                    == self.rewards.shape[0] == self.not_dones.shape[0] == self.not_dones_no_max.shape[0]
                    == self.env_rewards.shape[0])
        elif self._image_observations is not None and self.env_rewards is None:
            assert (self._observations.shape[0] - 1 == self._image_observations.shape[0] - 1 == self.actions.shape[0]
                    == self.rewards.shape[0] == self.not_dones.shape[0] == self.not_dones_no_max.shape[0])
        else:
            assert (self._observations.shape[0] - 1 == self._image_observations.shape[0] - 1 == self.actions.shape[0]
                    == self.rewards.shape[0] == self.not_dones.shape[0] == self.not_dones_no_max.shape[0]
                    == self.env_rewards.shape[0])
        if self.env_rewards is not None:
            assert self.rewards.shape == self.env_rewards.shape

    def __iter__(self) -> Transition:
        """
        Return each transition in the trajectory - one at a time

        Returns:
            state
            action
            reward
            goal reached
        """
        for step in range(len(self._observations) - 1):
            yield self[step]

    def __getitem__(self, index: t.Union[int, tuple]) -> t.Union["Trajectory", Transition]:
        """
        Return the transition stored at buffer position index

        Args:
            index: the transition or slice of transitions to grab. index is assumed to refer to the indices of the
            initial observations
        """
        if isinstance(index, tuple):
            start_indx, end_indx = index
            # the need to +1 to make sure we are grabbing the next observation for the observation at end_indx
            observations = self._observations[start_indx:end_indx + 1]
            image_observations = (self._image_observations[start_indx:end_indx + 1] if self._image_observations is not None else None)
            actions = self.actions[start_indx:end_indx]
            rewards = self.rewards[start_indx:end_indx]
            not_dones = self.not_dones[start_indx:end_indx]
            not_dones_no_max = self.not_dones_no_max[start_indx:end_indx]
            return Trajectory(observations, image_observations, actions=actions, rewards=rewards, not_dones=not_dones,
                              not_dones_no_max=not_dones_no_max,
                              env_rewards=self.env_rewards[start_indx:end_indx])

        else:
            return Transition(observation=self._observations[index],
                              image_observation=(self._image_observations[index]
                                                 if self._image_observations is not None else None),
                              action=self.actions[index],
                              next_observation=self._observations[index + 1],
                              next_image_observation=(self._image_observations[index+1]
                                                      if self._image_observations is not None else None),
                              reward=self.rewards[index], env_reward=self.env_rewards[index],
                              not_done=self.not_dones[index], not_done_no_max=self.not_dones_no_max[index])

    def __len__(self):
        """
        Return the number of transitions in the Trajectory
        """
        return len(self.actions)

    @property
    def initial_observations(self) -> np.ndarray:
        """
        The initial states in the trajectory
        """
        return self._observations[:-1]

    @property
    def initial_image_observations(self) -> t.Optional[np.ndarray]:
        """
        The initial image observations in the trajectory
        """
        return self._image_observations[:-1] if self._image_observations is not None else None

    @property
    def next_observations(self) -> np.ndarray:
        """
        The next states in the trajectory
        """
        return self._observations[1:]

    @property
    def next_image_observations(self) -> t.Optional[np.ndarray]:
        """
        The next image observations in the trajectory
        """
        return self._image_observations[1:] if self._image_observations is not None else None

    @property
    def all_observations(self) -> np.ndarray:
        """
        All states stored in the trajectory
        """
        return self._observations

    @property
    def all_image_observations(self) -> t.Optional[np.ndarray]:
        """
        All states stored in the trajectory
        """
        return self._image_observations if self._image_observations is not None else None

    @property
    def contains_done(self) -> bool:
        """
        Whether the trajectory contains the goal state
        """
        return np.sum(self.dones) == 1

    @property
    def total_reward(self) -> np.ndarray:
        """
        Compute the total reward received
        """
        return np.sum(self.rewards)

    @property
    def state_action_ndarray(self) -> np.ndarray:
        """
        The state action pairs through time
        """
        return np.concatenate([self.states[:-1], self.actions], axis=-1)

    def random_segment(self, length: int) -> "Trajectory":
        """
        Create a trajectory that is randomly select segment of this trajectory

        The segment is sampled such that there is a segment_length / num_transitions probability of
        sampling a segment that contains a goal state transition. As the target segment length approaches
        half the length of the trajectory, the goal state starts to be sampled with 50% probability.

        When an environment has only one goal state, all trajectory segments of a fixed length containing the
        trajectory's goal state are identical. This may have implications for how generalizable the learned
        reward function is.

        Args:
            length: the number of transitions to include in the segment

        Returns:
            contiguous subset of the trajectory
        """
        # this approach under samples trajectory segments containing the goal state
        # start_indx = np.random.randint(0, high=(self.init_states.shape[0] - length), size=1)[0]
        # this approach potentially over samples trajectory segments containing the goal state
        # pick a starting point anywhere in the trajectory
        start_indx = np.random.randint(0, high=self.initial_observations.shape[0])
        # if there are not length many transitions between the start index and the end of buffer,
        # move the start index
        if start_indx + length >= len(self):
            start_indx = len(self.actions) - length

        return self[(start_indx, start_indx + length)]

    def save(self, outpath: Path) -> None:
        """
        Save the trajectory as an npz to the specified location
        Args:
            outpath: location where the trajectory will be saved
        """
        np.savez(outpath.as_posix(),
                 states=self._observations, image_states=self._image_observations, actions=self.actions,
                 rewards=self.rewards,
                 not_dones=self.not_dones, not_dones_no_max=self.not_dones_no_max, env_rewards=self.env_rewards)

    @staticmethod
    def from_npz(in_path: Path) -> "Trajectory":
        """
        Load the trajectory from a npz archive
        Args:
            in_path: path the trajectory is stored in

        Returns:
            trajectory loaded from the specified file
        """
        assert (in_path.suffix == ".npz"
                and in_path.with_suffix(".npz").exists()), (f"{in_path.suffix} is not a valid file extension.\n"
                                                            f"Must be one of '.pt' or '.npz'")
        traj = np.load(in_path.as_posix())

        return Trajectory(observations=traj["states"], actions=traj["actions"], rewards=traj["rewards"],
                          not_dones=traj["not_dones"], not_dones_no_max=traj["not_dones_no_max"],
                          env_rewards=traj["env_rewards"], image_observations=traj["image_states"])


TRANSITION = t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, t.Union[float, np.ndarray], t.Union[float, np.ndarray], t.Optional[np.ndarray], t.Optional[np.ndarray], t.Optional[np.ndarray]]


class TrajectoryReplayBuffer:
    """
    Buffer to store trajectories of environment transitions. Unlike ReplayBuffer, which stores all transitions in a
    flat manner, transitions are sorted by trajectory. Each trajectory corresponds to an episode.
    """
    _RELABEL_BATCH_SIZE = 256

    def __init__(self, capacity: int, device: torch.device, window: int = 1, num_envs: t.Optional[int] = None,
                 image_observations: t.Optional[t.Union[int, np.ndarray]] = None):
        """
        Args:
            capacity: the number of trajectories to hold in memory
            device: the device sampled transitions should be put on
            window: no idea - part of the original code and is used in add_batch(...) which has not yet been refactored
            num_envs: the number of environment instances used to train the policy. Only needs to be specified when the
                      number is >1. Some algorithms train on multiple instances of an environment at once, e.g. PPO.
                      Not currently used, but not yet removed because we have not tested with an algorithm that needs
                      multiple environment instances.
            image_observations: (default = false) whether to collect image observations in addition to state
                                observations. This is helpful to use when the policy is trained on the state, but you
                                want to visualize the trajectories or the reward model is trained on images.

        """
        self.capacity = capacity
        self.device = device

        self.observations: t.Optional[np.ndarray] = None
        self.actions: t.Optional[np.ndarray] = None
        self.rewards: t.Optional[np.ndarray] = None
        self.not_dones: t.Optional[np.ndarray] = None
        self.not_dones_no_max: t.Optional[np.ndarray] = None
        self.trajectory_lengths: t.List = []
        self.window = window
        self.env_rewards: t.Optional[np.ndarray] = None
        self.image_observations: t.Optional[np.ndarray] = None
        # track whether to collect image observations - when not None, specifies the dimensions of the images
        self._collect_image_observations = image_observations

        # track the trajectories as a list of Trajectory
        self.trajectories: t.List[Trajectory] = []

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return np.sum(self.trajectory_lengths) - len(self.trajectory_lengths)

    def __getitem__(self, flat_indx: t.Union[int, t.Tuple[int, int], t.List[int]]) -> TRANSITION:
        """
        Get the transition at the given index

        Args:
            flat_indx: the index assuming transitions are stored flat instead of nested in trajectories
                        - when an integer is specified, a single transition is retrieved
                        - when a tuple of integers is given, a slice is retrieved as if the transitions are stored flat

        Returns:
             current observation
             action
             reward
             next observation
             whether the episode ended
             whether the episode ended without reaching max steps
             image version of current observation (optional)
        """
        if isinstance(flat_indx, int) or isinstance(flat_indx, np.int64):
            traj_indx, trans_indx = self._flat_indx_to_trajectory_index(flat_indx)
            # check we are grabbing from a trajectory currently being accumulated
            # When the done signal is given, the current trajectory being accumulated is converted to a trajectory,
            # is added to the list of trajectories, and the values used to accumulate the next trajectory are set to
            # done. The next trajectory is not started until the call to add(...) after the done signal is received.
            # Therefore, we need to check whether the trajectory to pull from is actually the last completed trajectory
            # prior to starting a new trajectory. This is why we compare the length of the lists containing trajectory
            # lengths and the list containing the trajectories.
            if (traj_indx == len(self.trajectory_lengths) - 1
                    and len(self.trajectory_lengths) > len(self.trajectories)):
                # we need to grab from the trajectory currently being populated
                return (self.observations[trans_indx].astype(np.float32), self.actions[trans_indx].astype(np.float32),
                        self.rewards[trans_indx].astype(np.float32), self.observations[trans_indx + 1].astype(np.float32),
                        self.not_dones[trans_indx].astype(np.float32),
                        self.not_dones_no_max[trans_indx].astype(np.float32),
                        (self.env_rewards[trans_indx].astype(np.float32)
                         if self.env_rewards is not None
                         else None),
                        ((self.image_observations[trans_indx].astype(np.float32))
                         if self.image_observations is not None
                         else None),
                        ((self.image_observations[trans_indx+1].astype(np.float32))
                         if self.image_observations is not None
                         else None))
            else:
                # grab from a previously completed trajectory
                transition: Transition = self.trajectories[traj_indx][trans_indx]
                return (transition.observation.astype(np.float32), transition.action.astype(np.float32),
                        transition.reward.astype(np.float32), transition.next_observation.astype(np.float32),
                        transition.not_done.astype(np.float32), transition.not_done_no_max.astype(np.float32),
                        transition.env_reward.astype(np.float32),
                        (transition.image_observation.astype(np.float32)
                         if transition.image_observation is not None
                         else None),
                        (transition.next_image_observation.astype(np.float32)
                         if transition.next_image_observation is not None
                         else None))
        elif isinstance(flat_indx, t.List):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            not_dones = []
            not_dones_no_max = []
            env_rewards = []
            image_observations = []
            next_image_observations = []
            for indx in flat_indx:
                observation, action, reward, next_observation, not_done, not_done_no_max, env_reward, image_observation, next_image_observation = self[indx]
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_observation)
                not_dones.append(not_done)
                not_dones_no_max.append(not_done_no_max)
                if env_reward is not None:
                    env_rewards.append(env_reward)
                if image_observation is not None:
                    image_observations.append(image_observation)
                if next_image_observation is not None:
                    next_image_observations.append(next_image_observation)
            return (np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32),
                    np.asarray(rewards, dtype=np.float32), np.asarray(next_observations, dtype=np.float32),
                    np.asarray(not_dones, dtype=np.float32), np.asarray(not_dones_no_max, dtype=np.float32),
                    (np.asarray(env_rewards, dtype=np.float32) if len(env_rewards) > 0 else None),
                    (np.asarray(image_observations, dtype=np.float32) if self._collect_image_observations else None),
                    (np.asarray(next_image_observations, dtype=np.float32) if self._collect_image_observations else None))
        else:
            # get the locations of the start and end transitions
            start_traj_indx, start_trans_indx = self._flat_indx_to_trajectory_index(flat_indx[0])
            end_traj_indx, end_trans_indx = self._flat_indx_to_trajectory_index(flat_indx[1])
            # check that we are not spanning trajectories
            if start_traj_indx == end_traj_indx:
                # grab the sub-trajectory
                sub_trajectory = self.trajectories[start_traj_indx][tuple((start_trans_indx, end_trans_indx))]
            else:
                # grab what remains of the trajectory
                end_trans_indx = len(self.trajectories[start_traj_indx]) - 1
                sub_trajectory = self.trajectories[start_traj_indx][tuple((start_trans_indx, end_trans_indx))]
            return (sub_trajectory.initial_observations,
                    sub_trajectory.actions,
                    sub_trajectory.rewards,
                    sub_trajectory.next_observations,
                    sub_trajectory.not_dones,
                    sub_trajectory.not_dones_no_max,
                    sub_trajectory.env_rewards,
                    (sub_trajectory.initial_image_observations
                     if sub_trajectory.initial_image_observations is not None
                     else None),
                    (sub_trajectory.next_image_observations
                     if sub_trajectory.next_image_observations is not None
                     else None))

    @property
    def trajectory_count(self) -> int:
        """
        The number of trajectories in the buffer
        """
        return len(self.trajectories)

    @property
    def all_not_dones(self) -> np.ndarray:
        """
        Rewards from the state-action pairs from all trajectories and all transitions, where the action was taken in the state
        """
        return np.concatenate([np.expand_dims(traj.not_dones, axis=0) for traj in self.trajectories], axis=0)

    @property
    def all_rewards(self) -> np.ndarray:
        """
        Rewards from the state-action pairs from all trajectories and all transitions, where the action was taken in the state
        """
        return np.concatenate([np.expand_dims(traj.rewards, axis=0) for traj in self.trajectories], axis=0)

    @property
    def all_environment_rewards(self) -> np.ndarray:
        """
        Environment rewards from all trajectories and all transitions
        """
        return np.concatenate([np.expand_dims(traj.rewards, axis=0) for traj in self.trajectories], axis=0)

    @property
    def all_initial_image_observations(self) -> np.ndarray:
        """
        Image observations from the state-action pairs from all trajectories and all transitions, where the action was taken in the state
        """
        return np.concatenate([np.expand_dims(traj.initial_image_observations, axis=0)
                               for traj in self.trajectories],
                              axis=0)

    @property
    def all_next_image_observations(self) -> np.ndarray:
        """
        Image observations from the state-action pairs from all trajectories and all transitions,

        The result of a transition
        """
        return np.concatenate([np.expand_dims(traj.next_image_observations, axis=0)
                               for traj in self.trajectories],
                              axis=0)

    @property
    def all_initial_observations(self) -> np.ndarray:
        """
        observations from the state-action pairs from all trajectories and all transitions, where the action was taken in the state
        """
        return np.concatenate([np.expand_dims(traj.initial_observations, axis=0) for traj in self.trajectories], axis=0)

    @property
    def all_next_observations(self) -> np.ndarray:
        """
        Observations from the state-action pairs from all trajectories and all transitions

        The result of a transition
        """
        return np.concatenate([np.expand_dims(traj.next_observations, axis=0) for traj in self.trajectories], axis=0)

    @property
    def all_actions(self) -> np.ndarray:
        """
        Actions from the state-action pairs from all trajectories and all transitions
        """
        return np.concatenate([np.expand_dims(traj.actions, axis=0) for traj in self.trajectories], axis=0)

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

    def _add_transition(self, observation: np.ndarray, action: np.ndarray, reward: float, done: t.Union[float, bool],
                        done_no_max: t.Union[float, bool],
                        env_reward: t.Optional[float] = None, image_observations: t.Optional[np.ndarray] = None):
        """
        Track the transition and update the length of the trajectory currently being accumulated

        Args:
            observation: the current observation
            action: the action taken in the current state
            reward: the reward associated with the last state-action pait
            done: whether the last action completed an episode
            done_no_max: whether the last action completed an episode without reaching the maximum allowed steps
            env_reward: (optional) the reward given by the environment - stored and used to train the preference-learned
                        reward model when learning from synthetic feedback
            image_observations: (optional) image-based observation -> should not be given is observations is also an image. This
                          should be used when you want to accumulate images separately from policy training.
        """
        self.observations = np.concatenate([self.observations, np.expand_dims(observation, axis=0)], axis=0)
        self.actions = np.concatenate([self.actions, np.expand_dims(action, axis=0)], axis=0)
        self.rewards = np.concatenate([self.rewards, np.asarray(reward).reshape(1, 1)], axis=0)
        if type(done) is float:
            self.not_dones = np.concatenate([self.not_dones,
                                             np.asarray(not done, dtype=np.float32).reshape(1, 1)], axis=0)
            self.not_dones_no_max = np.concatenate([self.not_dones_no_max,
                                                    np.asarray(not done_no_max, dtype=np.float32).reshape(1, 1)],
                                                   axis=0)
        else:
            self.not_dones = np.concatenate([self.not_dones,
                                             np.asarray(~done, dtype=np.float32).reshape(1, 1)], axis=0)
            self.not_dones_no_max = np.concatenate([self.not_dones_no_max,
                                                    np.asarray(~done_no_max, dtype=np.float32).reshape(1, 1)],
                                                   axis=0)

        self.trajectory_lengths[-1] += 1
        if env_reward is not None:
            self.env_rewards = np.concatenate([self.env_rewards,
                                               np.asarray(env_reward, dtype=np.float32).reshape(1, 1)], axis=0)

        if image_observations is not None and self._collect_image_observations:
            self.image_observations = np.concatenate([self.image_observations, np.expand_dims(image_observations, axis=0)], axis=0)

    def _start_trajectory(self, observation: np.ndarray,
                          action: np.ndarray,
                          reward: float,
                          done: t.Union[float, bool],
                          done_no_max: t.Union[float, bool],
                          env_reward: t.Optional[float] = None,
                          image_observations: t.Optional[np.ndarray] = None):
        """
        Start a new trajectory and track the transition

        Args:
            observation: the current observation
            action: the action taken in the current state
            reward: the reward associated with the last state-action pait
            done: whether the last action completed an episode
            done_no_max: whether the last action completed an episode without reaching the maximum allowed steps
            env_reward: (optional) the reward given by the environment - stored and used to train the preference-learned
                        reward model when learning from synthetic feedback
            image_observations: (optional) image-based observation -> should not be given is observations is also an image. This
                          should be used when you want to accumulate images separately from policy training.
        """
        self.observations = np.expand_dims(observation, axis=0).astype(dtype=np.float32)
        self.actions = np.expand_dims(action, axis=0).astype(dtype=np.float32)
        self.rewards = np.asarray(reward, dtype=np.float32).reshape(1, 1)
        if type(done) is float:
            self.not_dones = np.asarray(not done, dtype=np.float32).reshape(1, 1)
            self.not_dones_no_max = np.asarray(not done_no_max, dtype=np.float32).reshape(1, 1)
        else:
            self.not_dones = np.asarray(~done, dtype=np.float32).reshape(1, 1)
            self.not_dones_no_max = np.asarray(~done_no_max, dtype=np.float32).reshape(1, 1)

        self.trajectory_lengths.append(1)

        if env_reward is not None:
            self.env_rewards = np.asarray(env_reward, dtype=np.float32).reshape(1, 1)

        if image_observations is not None and self._collect_image_observations:
            self.image_observations = np.expand_dims(image_observations, axis=0).astype(dtype=np.float32)

    def add(self, observation, action, reward, next_observation, done, done_no_max,
            env_reward: t.Optional[float] = None, image_observation: t.Optional[np.ndarray] = None,
            image_next_observation: t.Optional[np.ndarray] = None):
        """
        Args:
            observation: the current observation
            action: the action taken in the current state
            reward: the reward associated with the last state-action pait
            next_observation: only used when an episode is completed to ensure the last observation is captured
            done: whether the last action completed an episode
            done_no_max: whether the last action completed an episode without reaching the maximum allowed steps
            env_reward: (optional) the reward given by the environment - stored and used to train the preference-learned
                        reward model when learning from synthetic feedback
            image_observation: (optional) image-based observation -> should not be given is observations is also an image. This
                        should be used when you want to accumulate images separately from policy training.
            image_next_observation: (optional) the image-based next observation -> should not be given when next_observation is also
                            and image. This should be used when you want to accumulate the images separately from the
                            trained policy.
        """
        if self.observations is None:
            self._start_trajectory(observation, action, reward, done, done_no_max, env_reward, image_observation)
        elif done:
            self._add_transition(observation, action, reward, done, done_no_max, env_reward, image_observation)
            # the episode has ended, so we need to track the next observation
            self.observations = np.concatenate([self.observations, np.expand_dims(next_observation, axis=0)], axis=0)
            if image_next_observation is not None:
                self.image_observations = np.concatenate([self.image_observations,
                                                          np.expand_dims(image_next_observation, axis=0)], axis=0)
            # create the trajectory
            self.trajectories.append(Trajectory(self.observations.astype(dtype=np.float32),
                                                (self.image_observations.astype(dtype=np.float32)
                                                 if self.image_observations is not None
                                                 else None),
                                                actions=self.actions.astype(dtype=np.float32),
                                                rewards=self.rewards.astype(dtype=np.float32),
                                                not_dones=self.not_dones.astype(dtype=np.float32),
                                                not_dones_no_max=self.not_dones_no_max.astype(dtype=np.float32),
                                                env_rewards=self.env_rewards.astype(dtype=np.float32)))
            # check if the inclusion of the just completed trajectory puts the buffer at capacity
            # if it does, remove the first trajectory as this is a FIFO buffer
            if np.sum(self.trajectory_lengths) >= self.capacity:
                self.trajectories = self.trajectories[1:]
                self.trajectory_lengths = self.trajectory_lengths[1:]
            self.observations = None
            self.actions = None
            self.rewards = None
            self.not_dones = None
            self.not_dones_no_max = None
            self.env_rewards = None
            self.image_observations = None
        else:
            self._add_transition(observation, action, reward, done, done_no_max, env_reward, image_observation)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def relabel_with_predictor(self, predictor, state_action_formatter: PreProcessInference):
        """
        Relabel the rewards stored in the replay buffer using the given predictor

        Args:
            predictor: network that will consume state-action pairs and assign a reward
            state_action_formatter: formats the states and actions for consumption by the reward model
        """
        print("Relabelling the replay buffer with the updated reward model.")
        for trajectory in self.trajectories:
            # the number of batches to run through the model
            total_iter = int(len(trajectory) / self._RELABEL_BATCH_SIZE)
            # handle the case where we have more transitions than is evenly divisible by the batch size
            if len(trajectory) > self._RELABEL_BATCH_SIZE * total_iter:
                total_iter += 1
            # collect and process each batch to be passed through predictor
            for index in range(total_iter):
                start_indx = index * self._RELABEL_BATCH_SIZE
                # make sure we don't have an end index that is after the end of the trajectory
                end_indx = min((index + 1) * self._RELABEL_BATCH_SIZE, len(trajectory))

                # pull out the actions from the transitions that will be relabelled
                actions = trajectory.actions[start_indx:end_indx]
                # we need to handle the case where the reward model operates off of images
                if predictor.image_observations:
                    observations = trajectory.all_image_observations[start_indx:end_indx]
                else:
                    observations = trajectory.all_observations[start_indx:end_indx]
                formatted_state_action = state_action_formatter.format_state_action(observations, actions, batch_sa=True)
                pred_reward = predictor.r_hat_batch(formatted_state_action)
                # update the rewards assigned to the transitions
                trajectory.rewards[start_indx:end_indx] = pred_reward

    def sample(self, batch_size: int):
        indxs = list(np.random.randint(0, np.sum(self.trajectory_lengths) - 1, size=batch_size))
        observations, actions, rewards, next_observations, not_dones, not_dones_no_max, env_rewards, image_observations, next_image_observations = self[indxs]
        observations = torch.as_tensor(observations, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_observations = torch.as_tensor(next_observations, device=self.device).float()
        not_dones = torch.as_tensor(not_dones, device=self.device)
        not_dones_no_max = torch.as_tensor(not_dones_no_max, device=self.device)
        env_rewards = torch.as_tensor(env_rewards, device=self.device)
        image_observations = (torch.as_tensor(image_observations, device=self.device).float() if self._collect_image_observations else None)
        next_image_observations = (torch.as_tensor(next_image_observations, device=self.device).float() if self._collect_image_observations else None)
        return observations, actions, rewards, next_observations, not_dones, not_dones_no_max, env_rewards, image_observations, next_image_observations

    def sample_state_ent(self, batch_size: int):
        observations, actions, rewards, next_observations, not_dones, not_dones_no_max, _, _, _ = self.sample(batch_size)
        full_observation = torch.as_tensor(np.concatenate([traj.all_observations for traj in self.trajectories], axis=0),
                                           device=self.device)
        return observations, full_observation, actions, rewards, next_observations, not_dones, not_dones_no_max

    def save(self, out_directory: Path, env_id: str, step: int):
        """
        Save the replay buffer to disk as a npz archive
        Args:
            out_directory: location where replay buffer will be saved
            env_id: the environment within which the data was  generated
            step: the number of policy training steps taken to produce this dataset
        """
        # create the ZipFile object
        zip_obj = ZipFile(out_directory / f"{env_id}_replay_buffer_{step}.zip", "w")

        # write each trajectory file to disk and to the zip archive
        for traj_id, trajectory in enumerate(self.trajectories):
            trajectory.save(out_directory / f"{traj_id}.npz")
            zip_obj.write(out_directory / f"{traj_id}.npz")
        # close the Zip File
        zip_obj.close()

    @staticmethod
    def from_directory(directory_path: Path,
                       device: torch.device = 'cuda') -> "TrajectoryReplayBuffer":
        """
        Create a TrajectoryReplay buffer from a directory of npz archive trajectories

        Args:
            directory_path: the location of the npz_archive on disk
            device: the device sampled transitions should be pushed to
        Returns:
            populated trajectory replay buffer
        """
        # accumulate the trajectories
        trajectories = []
        trajectory_lengths = []
        # determine how many transitions are in the replay buffer
        capacity = 0
        # load each trajectory from disk
        for traj_filename in directory_path.iterdir():
            # we only load data from npz archives, so we need to skip anything else
            if not traj_filename.suffix == ".npz": continue
            # load the trajectory from disk
            traj = Trajectory.from_npz(traj_filename)
            # track the trajectory
            trajectories.append(traj)
            # track the trajectory's length
            trajectory_lengths.append(len(traj))
            # track the trajectory's length
            capacity += len(traj)
        # create the buffer
        _buffer = TrajectoryReplayBuffer(capacity=capacity, device=device)
        # add the trajectories to the buffer
        _buffer.trajectories = trajectories
        _buffer.trajectory_lengths = trajectory_lengths

        return _buffer


class ReplayBuffer:
    """Buffer to store environment transitions."""
    def __init__(self, observation_shape, action_shape, capacity, device, window=1, num_envs=None,
                 image_observations: t.Optional[t.List[int]] = None):
        self.capacity = capacity
        self.device = device

        # the proprioceptive observation is stored as float32, pixels observation as uint8
        observation_dtype = np.float32 if len(observation_shape) == 1 else np.uint8

        self.observations = (np.empty((capacity, *observation_shape), dtype=observation_dtype) if num_envs is None
                      else np.empty((capacity, num_envs, *observation_shape), dtype=observation_dtype))
        self.next_observations = (np.empty((capacity, *observation_shape), dtype=observation_dtype) if num_envs is None
                           else np.empty((capacity, num_envs, *observation_shape), dtype=observation_dtype))
        self.actions = (np.empty((capacity, *action_shape), dtype=np.float32) if num_envs is None
                        else np.empty((capacity, num_envs, *action_shape), dtype=observation_dtype))
        self.rewards = (np.empty((capacity, 1), dtype=np.float32) if num_envs is None
                        else np.empty((capacity, num_envs, 1), dtype=observation_dtype))
        self.not_dones = (np.empty((capacity, 1), dtype=np.float32) if num_envs is None
                          else np.empty((capacity, num_envs, 1), dtype=observation_dtype))
        self.not_dones_no_max = (np.empty((capacity, 1), dtype=np.float32) if num_envs is None
                                 else np.empty((capacity, num_envs, 1), dtype=observation_dtype))
        self.window = window
        if image_observations is None:
            self.image_observations = None
        else:
            self.image_observations = (np.empty((capacity, *image_observations), dtype=np.float32)  if num_envs is None
                                else np.empty((capacity, num_envs, *image_observations), dtype=np.float32))

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, observation, action, reward, next_observation, done, done_no_max,
            image_observations: t.Optional[np.ndarray] = None):
        np.copyto(self.observations[self.idx], observation)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_observations[self.idx], next_observation)
        if image_observations is not None and self.image_observations is not None:
            np.copyto(self.image_observations[self.idx], image_observations)
        if type(done) is float:
            np.copyto(self.not_dones[self.idx], not done)
            np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        else:
            np.copyto(self.not_dones[self.idx], ~done)
            np.copyto(self.not_dones_no_max[self.idx], ~done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, observation, action, reward, next_observation, done, done_no_max,
            image_observations: t.Optional[np.ndarray] = None):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.observations[self.idx:self.capacity], observation[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_observations[self.idx:self.capacity], next_observation[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            if image_observations is not None and self.image_observations is not None:
                np.copyto(self.image_observations[self.idx:self.capacity], image_observations[:maximum_index])
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.observations[0:remain], observation[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_observations[0:remain], next_observation[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
                if image_observations is not None and self.image_observations is not None:
                    np.copyto(self.image_observations[0:remain], image_observations[maximum_index:])
            self.idx = remain
        else:
            np.copyto(self.observations[self.idx:next_index], observation)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_observations[self.idx:next_index], next_observation)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            if image_observations is not None and self.image_observations is not None:
                np.copyto(self.image_observations[self.idx:next_index], image_observations)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx

            actions = self.actions[index*batch_size:last_index]
            # inputs = np.concatenate([observations, actions], axis=-1)
            # pred_reward = predictor.r_hat_batch(inputs)
            if predictor.image_observations:
                image_observations = self.image_observations[index * batch_size:last_index]
                pred_reward = predictor.r_hat(predictor.format_state_action(image_observations, actions, batch_sa=True))
            else:
                observations = self.observations[index * batch_size:last_index]
                pred_reward = predictor.r_hat(predictor.format_state_action(observations, actions, batch_sa=True))
            self.rewards[index*batch_size:last_index] = pred_reward
            
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        observations = torch.as_tensor(self.observations[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_observations = torch.as_tensor(self.next_observations[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return observations, actions, rewards, next_observations, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        observations = torch.as_tensor(self.observations[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_observations = torch.as_tensor(self.next_observations[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_observation = self.observations
        else:
            full_observation = self.observations[: self.idx]
        full_observation = torch.as_tensor(full_observation, device=self.device)
        
        return observations, full_observation, actions, rewards, next_observations, not_dones, not_dones_no_max

    def save(self, outpath: str):
        """
        Save the replay buffer to disk as a npz archive
        Args:
            outpath: location where replay buffer will be saved
        """
        if self.image_observations is None:
            np.savez(outpath,
                     observations=self.observations,
                     actions=self.actions,
                     next_observations=self.next_observations,
                     rewards=self.rewards,
                     dones=self.not_dones,
                     dones_no_max=self.not_dones_no_max,
                     idx=self.idx)
        else:
            np.savez(outpath,
                     observations=self.observations,
                     image_observations=self.image_observations,
                     actions=self.actions,
                     next_observations=self.next_observations,
                     rewards=self.rewards,
                     dones=self.not_dones,
                     dones_no_max=self.not_dones_no_max,
                     idx=self.idx)

    @staticmethod
    def from_npz_archive(npz_path: str, device='cuda', capacity: t.Optional[int] = None) -> "ReplayBuffer":
        """
        Create a replay buffer from the npz archive at the given path
        Args:
            npz_path: the location of the npz_archive on disk

        Returns:

        """
        npz_archive = np.load(npz_path)
        observations = npz_archive["observations"]
        actions = npz_archive["actions"]
        next_observations = npz_archive["next_observations"]
        rewards = npz_archive["rewards"]
        dones = npz_archive["dones"]
        dones_no_max = npz_archive["dones_no_max"]

        _buffer = ReplayBuffer(observation_shape=observations.shape[1:],
                               action_shape=actions.shape[1:],
                               capacity=(observations.shape[0] if capacity is None else capacity),
                               device=device)

        _buffer.observations[:observations.shape[0]] = observations
        _buffer.actions[:actions.shape[0]] = actions
        _buffer.next_observations[:next_observations.shape[0]] = next_observations
        _buffer.rewards[:rewards.shape[0]] = rewards
        _buffer.not_dones[:dones.shape[0]] = dones
        _buffer.not_dones_no_max[:dones_no_max.shape[0]] = dones_no_max
        not_populated_indxs = np.where(np.all((_buffer.observations == 0), axis=-1))[0]
        if len(not_populated_indxs) > 0:
            _buffer.full = False
            _buffer.idx = not_populated_indxs[0]
        else:
            _buffer.full = (capacity == observations.shape[0])
            _buffer.idx = observations.shape[0]

        return _buffer
