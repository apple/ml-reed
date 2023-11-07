#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import typing as t
from pathlib import Path
import time

import numpy as np
import torch

from omegaconf import dictconfig, OmegaConf
import hydra

from BPref import utils
from BPref.logger import Logger
from BPref.replay_buffer import TrajectoryReplayBuffer
from collections import deque

from reed.models.reward_model import StateActionRewardModel

from reed.data.preference_dataset import PreferenceDataset
from reed.data.preference_data_loader import PreferenceTripletEnsembleDataLoader
from reed.data.preprocess_images import PreProcessInference


class PEBBLE:
    """
    Train a reward model in conjunction with policy training following the PEBBLE algorithm from (Lee et al. 2021)
    """

    def __init__(self, experiment_config: dictconfig.DictConfig):
        """
        Args:
            experiment_config: contains the configuration for the experiment to be run. Access like a dictionry
        """
        # track the experimental configuration
        self.experiment_config = experiment_config

        # create the logger to track policy learning progress
        self.logger = Logger(
            self.experiment_config.out_dir,
            save_tb=self.experiment_config.log_save_tb,
            log_frequency=self.experiment_config.log_frequency,
            agent=self.experiment_config.agent.name)

        # used to track where we are in training
        # total amount of feedback the reward model has solicited
        self.total_feedback = 0
        # total amount of feedback given to the reward model
        self.labeled_feedback = 0
        # policy train step
        self.step = 0

        # we need to set the random seed for replication purposes
        utils.set_seed_everywhere(self.experiment_config.seed)

        # the device on which models will be trained
        self.device = torch.device(self.experiment_config.device)
        # flag to make sure we are handling multi-gpu training where we need to
        self.multi_gpu = torch.cuda.device_count() > 1
        print("----------------------------------------")
        print("----------------------------------------")
        print("----------------------------------------")
        print("----------------------------------------")
        print(f"There is {torch.cuda.device_count()} GPU, so models will be trained with torch.nn.DataParallel.")
        print("----------------------------------------")
        print("----------------------------------------")
        print("----------------------------------------")
        print("----------------------------------------")

        # make the environment
        if 'metaworld' in self.experiment_config.env:
            self.env = utils.make_metaworld_env(self.experiment_config)
            # we are not evaluating a domain where we need to log whether an agent has reached a goal state
            self.log_success = True
        else:
            self.env = utils.make_env(self.experiment_config)
            # we are not evaluating a domain where we need to log whether an agent has reached a goal state
            self.log_success = False
        print('----------------------')
        print('----------------------')
        print('----------------------')
        print('----------------------')
        print("observation space ", self.env.observation_space.shape[0])
        print("action space ", self.env.action_space.shape[0])
        print('----------------------')
        print('----------------------')
        print('----------------------')
        print('----------------------')
        # we need to set the policy's observation and action space
        self.experiment_config.agent.params.obs_dim = self.env.observation_space.shape[0]
        self.experiment_config.agent.params.action_dim = self.env.action_space.shape[0]
        self.experiment_config.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        # create the agent specified in the configuration
        self.agent = hydra.utils.instantiate(self.experiment_config.agent)

        # the class that will format the observations and observation action pairs for consumption by the reward model
        self._reward_input_preprocessor = PreProcessInference(
            image_observations=self.experiment_config.reward_from_image_observations,
            grayscale_images=self.experiment_config.grayscale_images,
            normalize_images=self.experiment_config.normalized_images)

        # determine the reward's observation space
        # if the reward is trained on images then the reward's observation space differs from the policy's, which is
        # trained on the state space
        self._observation_dimensionality = self._determine_observation_dimensions()
        self._reward_observation_dimensionality = self._determine_reward_observation_dimensions()

        # create the agent's replay buffer setting if image observations will need to be tracked
        self.replay_buffer = TrajectoryReplayBuffer(
            int(self.experiment_config.replay_buffer_capacity),
            self.device,
            image_observations=(self._observation_dimensionality
                                if (self.experiment_config.reward_from_image_observations
                                    or self.experiment_config.save_image_observations)
                                else None)
        )

        # determine the dimensionality of the input to the reward function
        self.reward_in_dim = self._determine_reward_input_dimensions(
            observation_dim=self._reward_observation_dimensionality,
            action_dim=self.env.action_space.shape[0])

        # instantiating the reward model
        self.reward_model = self.construct_reward_ensemble()

        # create the preference dataset that will solicit and hold labelled preference triplets
        self.preference_dataset = PreferenceDataset(
            observation_dim=self._reward_observation_dimensionality,
            action_dim=self.env.action_space.shape[0],
            capacity=self.experiment_config.preference_dataset_capacity,
            size_segment=self.experiment_config.segment_size,
            out_path=Path("/tmp/preference_dataset/"),
            image_observations=self.experiment_config.reward_from_image_observations,
            state_action_formatter=self._reward_input_preprocessor,
            grayscale_images=self.experiment_config.grayscale_images,
            collect_image_pref_dataset=self.experiment_config.save_image_observations,
            teacher_beta=self.experiment_config.teacher_beta,
            teacher_gamma=self.experiment_config.teacher_gamma,
            teacher_eps_mistake=self.experiment_config.teacher_eps_mistake,
            teacher_eps_skip=self.experiment_config.teacher_eps_skip,
            teacher_eps_equal=self.experiment_config.teacher_eps_equal
        )

        # save the experimental configuration
        with open(Path(self.experiment_config.out_dir) / "experiment_config.yaml", "w+") as f:
            OmegaConf.save(config=self.experiment_config, f=f)

    def _determine_reward_input_dimensions(self,
                                           observation_dim: t.Union[int, np.ndarray],
                                           action_dim: int) -> t.Union[int, t.Sequence]:
        """
        Determine the dimensionality of the inputs to the reward model

        Args:
            observation_dim: the dimensionality of agent observations. If the observation is an image, the
                             dimensionality should have the following order: (num_channels, height, width)
            action_dim: the dimensionality of agent actions
        Returns:
            the dimensionality of the reward model's inputs
        """
        # compute the dimensions of the input to the reward function
        if not self.experiment_config.reward_from_image_observations:
            return observation_dim + action_dim
        else:
            # we need to concatenate the actions to last dimension of the image
            # the input to the reward net also needs to have the channels first
            # the image dimensions are given to us a (height, width, channels)
            sample_shape = list(observation_dim)
            if self.experiment_config.grayscale_images:
                num_channels = action_dim + 1
            else:
                num_channels = sample_shape[0] + action_dim
            # update the number of channels
            sample_shape[0] = num_channels
            # the dimensions of the input to the reward model
            return sample_shape

    def _determine_reward_observation_dimensions(self) -> t.Union[int, np.ndarray]:
        """
        Check if the reward will use the image observations.

        If so the reward input shape needs to be set accordingly

        Returns:
            the dimensionality of reward's observation space
        """
        if self.experiment_config.reward_from_image_observations:
            # get a sample image rendering of the environment and get its shape
            self.env.reset()
            if "metaworld" in self.experiment_config.env:
                start_time = time.time()
                img_obs = self.env.render(camera_name=self.experiment_config.camera_name,
                                          resolution=(
                                          self.experiment_config.image_height, self.experiment_config.image_width))
                end_time = time.time()
                print(f"Sample render time for metaworld is {end_time - start_time} seconds")
            else:
                start_time = time.time()
                img_obs = self.env.render(mode="rgb_array",
                                          height=self.experiment_config.image_height,
                                          width=self.experiment_config.image_width)
                end_time = time.time()
                print(f"Sample render time for DMC is {end_time - start_time} seconds")
            formatted_image_observation = self._reward_input_preprocessor.format_state(img_obs).squeeze(axis=0)
            observation_space = formatted_image_observation.shape
            print("--------------------------")
            print("--------------------------")
            print("--------------------------")
            print("image observation shape", observation_space)
            print("--------------------------")
            print("--------------------------")
            print("--------------------------")
        else:
            observation_space = self.env.observation_space.shape[0]

        return observation_space

    def _determine_observation_dimensions(self) -> t.Union[int, np.ndarray]:
        """
        Check if the reward will use the image observations.

        If so the replay buffer needs to be set up to
        accumulate the image observations

        Returns:
            the dimensionality of reward's observation space
        """
        if self.experiment_config.reward_from_image_observations:
            # get a sample image rendering of the environment and get its shape
            self.env.reset()
            if "metaworld" in self.experiment_config.env:
                start_time = time.time()
                img_obs = self.env.render(camera_name=self.experiment_config.camera_name,
                                          resolution=(
                                          self.experiment_config.image_height, self.experiment_config.image_width))
                end_time = time.time()
                print(f"Sample render time for metaworld is {end_time - start_time} seconds")
            else:
                start_time = time.time()
                img_obs = self.env.render(mode="rgb_array",
                                          height=self.experiment_config.image_height,
                                          width=self.experiment_config.image_width)
                end_time = time.time()
                print(f"Sample render time for DMC is {end_time - start_time} seconds")
            observation_space = img_obs.shape
            print("--------------------------")
            print("--------------------------")
            print("--------------------------")
            print("image observation shape", observation_space)
            print("--------------------------")
            print("--------------------------")
            print("--------------------------")
        else:
            observation_space = self.env.observation_space.shape[0]

        return observation_space

    def _render_image_observation(self) -> np.ndarray:
        """
        Render the current image observation
        """
        if "metaworld" in self.experiment_config.env:
            img_obs = self.env.render(camera_name=self.experiment_config.camera_name,
                                      resolution=(
                                      self.experiment_config.image_height, self.experiment_config.image_width))
        else:
            img_obs = self.env.render(mode="rgb_array",
                                      height=self.experiment_config.image_height,
                                      width=self.experiment_config.image_width)
        return img_obs

    def construct_reward_ensemble(self) -> StateActionRewardModel:
        """
        Create the reward ensemble as specified in the experiment config.
        """
        return StateActionRewardModel(
            in_dim=self.reward_in_dim,
            ensemble_size=self.experiment_config.ensemble_size,
            hidden_dim=self.experiment_config.reward_hidden_embed_dim,
            hidden_layers=self.experiment_config.reward_num_hidden_layers,
            final_activation=self.experiment_config.activation,
            lr=self.experiment_config.reward_lr,
            reward_train_batch=self.experiment_config.reward_train_batch,
            size_segment=self.experiment_config.segment_size,
            device=self.device,
            multi_gpu=self.multi_gpu,
            image_observations=self.experiment_config.reward_from_image_observations,
            image_encoder_architecture=self.experiment_config.image_encoder_architecture,
            image_hidden_num_channels=self.experiment_config.image_hidden_num_channels,
            grayscale_images=self.experiment_config.grayscale_images
        )

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        for episode in range(self.experiment_config.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.experiment_config.num_eval_episodes
        average_true_episode_reward /= self.experiment_config.num_eval_episodes
        if self.log_success:
            success_rate /= self.experiment_config.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)
        self.logger.dump(self.step)

    def train_reward_on_preferences(self) -> t.Optional[float]:
        """
        Update the reward model on the current preference dataset

        Returns:
            train accuracy on the current reward model update round, if the preference dataset contains samples
        """
        # create the data loader that will be used to train the reward model
        preference_data_loader = PreferenceTripletEnsembleDataLoader(
            self.preference_dataset,
            ensemble_size=self.experiment_config.ensemble_size,
            batch_size=self.experiment_config.reward_train_batch,
            num_workers=4)
        print("training the reward model!!!")
        if self.labeled_feedback > 0:
            # update reward
            if self.experiment_config.label_margin > 0 or self.experiment_config.teacher_eps_equal > 0:
                train_accuracies = self.reward_model.train_soft_reward(preference_data_loader=preference_data_loader,
                                                                       num_epoch=self.experiment_config.reward_update)
            else:
                train_accuracies = self.reward_model.train_reward(preference_data_loader=preference_data_loader,
                                                                  num_epoch=self.experiment_config.reward_update)

            # save the reward model in its current state
            self.reward_model.save(self.experiment_config.out_dir, env_id=self.experiment_config.env, step=self.step)

            return float(np.mean(train_accuracies))
        else:
            return None

    def grow_preference_dataset(self, first_flag: bool = False):
        """
        Grow the preference feedback by soliciting feedback one queries selected according to the specified
        sampling method
        """
        # corner case: new total feed > max feed
        if (
                self.experiment_config.preference_dataset_update_size + self.total_feedback) > self.experiment_config.max_feedback:
            mb_size = self.experiment_config.max_feedback - self.total_feedback
        else:
            mb_size = self.experiment_config.preference_dataset_update_size

        if first_flag:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.preference_dataset.uniform_sampling(
                self.replay_buffer,
                mb_size=mb_size)
        else:
            if self.experiment_config.feed_type == 0:
                labeled_queries = self.preference_dataset.uniform_sampling(
                    self.replay_buffer,
                    mb_size=mb_size)
            elif self.experiment_config.feed_type == 1:
                labeled_queries = self.preference_dataset.disagreement_sampling(
                    self.replay_buffer,
                    mb_size=mb_size,
                    large_batch=self.experiment_config.preference_dataset_large_update_size,
                    reward_model=self.reward_model)

        # we need to track how much feedback has been solicited and how much has been labelled
        self.total_feedback += self.experiment_config.preference_dataset_update_size
        self.labeled_feedback += labeled_queries

    def update_reward(self, first_flag: bool = False):
        """
        Update the preference dataset and train the reward model
        """
        # grow the preference dataset
        self.grow_preference_dataset(first_flag=first_flag)

        # train the reward model on the updated preference dataset
        train_accuracy = self.train_reward_on_preferences()

        print(f"Reward function is updated!! ACC: {train_accuracy}")

    def save_preference_dataset(self):
        pref_dataset_out_dir = Path(
            self.experiment_config.out_dir) / f"{self.experiment_config.env}_preference_dataset_{self.step}"
        pref_dataset_out_dir.mkdir(parents=True, exist_ok=True)
        print(pref_dataset_out_dir, pref_dataset_out_dir.exists())
        self.preference_dataset.save(pref_dataset_out_dir, env_id=self.experiment_config.env, step=self.step)
        print(f"Preference dataset saved to {pref_dataset_out_dir}")

    def save_replay_buffer(self):
        replay_out_dir = Path(
            self.experiment_config.out_dir) / f"{self.experiment_config.env}_replay_buffer_{self.step}"
        replay_out_dir.mkdir(parents=True, exist_ok=True)
        print(replay_out_dir, replay_out_dir.exists())
        self.replay_buffer.save(out_directory=replay_out_dir, env_id=self.experiment_config.env, step=self.step)
        print(f"Replay buffer saved to {replay_out_dir}")

    def save_everything(self):
        self.agent.save(self.experiment_config.out_dir, env_id=self.experiment_config.env, step=self.step)
        self.reward_model.save(self.experiment_config.out_dir, env_id=self.experiment_config.env, step=self.step)
        print("Agent and reward models saved to: ", self.experiment_config.out_dir)
        # save anything in the log that has yet to be saved
        self.logger.dump(self.step, save=True)

        # save the preference dataset and the replay buffer
        self.save_preference_dataset()
        self.save_replay_buffer()

    def run(self):
        print("Starting training......")
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        obs = None
        img_obs = None
        next_obs = None
        img_next_obs = None

        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        interact_count = 0
        while self.step < self.experiment_config.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.experiment_config.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.experiment_config.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                obs = self.env.reset()
                if self.experiment_config.reward_from_image_observations or self.experiment_config.save_image_observations:
                    img_obs = self._render_image_observation()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.experiment_config.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step == (self.experiment_config.num_seed_steps + self.experiment_config.num_unsup_steps):
                print("----------------------------------------------")
                print("----------------------------------------------")
                print("----------------------------------------------")
                print(f"Updating the reward model for the first time: step = {self.step}")
                print("----------------------------------------------")
                print("----------------------------------------------")
                print("----------------------------------------------")
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (
                            self.experiment_config.segment_size / self.env._max_episode_steps)
                self.preference_dataset.set_teacher_thres_skip(new_margin)
                self.preference_dataset.set_teacher_thres_equal(new_margin)

                # collect more preference feedback and update the reward
                self.update_reward(first_flag=True)

                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model,
                                                          state_action_formatter=self._reward_input_preprocessor)

                # reset Q due to unsupervised exploration
                self.agent.reset_critic()

                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.experiment_config.reset_update,
                    policy_update=True)

                # reset interact_count
                interact_count = 0
            elif self.step > self.experiment_config.num_seed_steps + self.experiment_config.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.experiment_config.max_feedback:
                    if interact_count == self.experiment_config.num_interact:
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (
                                    self.experiment_config.segment_size / self.env._max_episode_steps)
                        self.preference_dataset.set_teacher_thres_skip(
                            new_margin * self.experiment_config.teacher_eps_skip)
                        self.preference_dataset.set_teacher_thres_equal(
                            new_margin * self.experiment_config.teacher_eps_equal)

                        self.update_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model,
                                                                  state_action_formatter=self._reward_input_preprocessor)
                        # we need to reset the counter tracking the number of interactions the agent has had with the
                        # environment between reward model updates
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
            # unsupervised exploration
            elif self.step > self.experiment_config.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.experiment_config.topK)

            next_obs, reward, done, extra = self.env.step(action)
            # render the current environment if the reward function is using image observations or we are collecting the
            # image-based observations so save
            if self.experiment_config.reward_from_image_observations or self.experiment_config.save_image_observations:
                img_next_obs = self._render_image_observation()
            # get the reward value we are learning from preferences
            if self.experiment_config.reward_from_image_observations:
                reward_hat = self.reward_model.r_hat(self._reward_input_preprocessor.format_state_action(img_obs,
                                                                                                         action))
            else:
                reward_hat = self.reward_model.r_hat(self._reward_input_preprocessor.format_state_action(obs, action))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            # adding data to the replay buffer and reward training data
            if self.experiment_config.reward_from_image_observations or self.experiment_config.save_image_observations:
                self.replay_buffer.add(
                    obs, action, reward_hat,
                    next_obs, done, done_no_max, env_reward=reward,
                    image_observation=img_obs, image_next_observation=img_next_obs)
            else:
                self.replay_buffer.add(
                    obs, action, reward_hat, next_obs, done, done_no_max, env_reward=reward)

            obs = next_obs
            # render the current environment if the reward function is using image observations
            if self.experiment_config.reward_from_image_observations or self.experiment_config.save_image_observations:
                img_obs = np.copy(img_next_obs)
            episode_step += 1
            self.step += 1
            interact_count += 1

        # save everything involved in this experimental run
        self.save_everything()
