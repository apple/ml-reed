#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from pathlib import Path
from omegaconf import OmegaConf
import turibolt as bolt

from logger import Logger
from replay_buffer import TrajectoryReplayBuffer
from reward_model import RewardModel
from collections import deque

import utils
import hydra


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        print("line 33", self.cfg)
        self.logger = Logger(
            # self.work_dir,
            bolt.ARTIFACT_DIR,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        print(self.device)
        self.log_success = False

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
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)
        print('----------------------')
        print('----------------------')
        print('----------------------')
        print('----------------------')
        print("observation space", self.env.observation_space.shape[0])
        print("action space", self.env.action_space.shape[0])
        print('----------------------')
        print('----------------------')
        print('----------------------')
        print('----------------------')
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # check if the reward will use the image observations
        # if so the reward input shape needs to be set accordingly and the replay buffer needs to be set up to
        # accumulate the image observations
        if cfg.reward_from_image_observations:
            # get a sample image rendering of the environment and get its shape
            self.env.reset()
            if "metaworld" in self.cfg.env:
                start_time = time.time()
                img_obs = self.env.render(camera_name=self.cfg.camera_name,
                                          resolution=(self.cfg.image_height, self.cfg.image_width))
                end_time = time.time()
                print(f"Sample render time for metaworld is {end_time - start_time} seconds")
            else:
                start_time = time.time()
                img_obs = self.env.render(mode="rgb_array",
                                               height=self.cfg.image_height,
                                               width=self.cfg.image_width)
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

        # create the agent's replay buffer setting if image observations will need to be tracked
        self.replay_buffer = TrajectoryReplayBuffer(
            int(cfg.replay_buffer_capacity),
            self.device,
            image_observations=(observation_space
                                if cfg.reward_from_image_observations or cfg.save_image_observations
                                else None)
        )
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            observation_space,
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            reward_train_batch=cfg.reward_train_batch,
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            image_observations=cfg.reward_from_image_observations,
            image_encoder_architecture=cfg.image_encoder_architecture,
            image_hidden_num_channels=cfg.image_hidden_num_channels,
            collect_image_pref_dataset=cfg.save_image_observations,
            experience_buffer=self.replay_buffer,
            device=self.device,
            multi_gpu=self.multi_gpu,
            grayscale_images=cfg.grayscale_images
        )

        if self.cfg.reward_from_pretrained:
            self.reward_model.from_pretrained(blobby_artifact_uri=self.cfg.blobby_artifact_uri)

        # save the experimental configuration
        with open(Path(bolt.ARTIFACT_DIR) / "experiment_config.yaml", "w+") as f:
            # yaml.dump(self.cfg, f)
            OmegaConf.save(config=self.cfg, f=f)

    def _render_image_observation(self) -> np.ndarray:
        """
        Render the current image observation
        """
        if "metaworld" in self.cfg.env:
            start_time = time.time()
            img_obs = self.env.render(camera_name=self.cfg.camera_name,
                                      resolution=(self.cfg.image_height, self.cfg.image_width))
            end_time = time.time()
            bolt.send_metrics({"image_render_seconds": (end_time - start_time)})
        else:
            start_time = time.time()
            img_obs = self.env.render(mode="rgb_array",
                                      height=self.cfg.image_height,
                                      width=self.cfg.image_width)
            end_time = time.time()
            bolt.send_metrics({"image_render_seconds": (end_time - start_time)})
        return img_obs

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
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
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        print("training the reward model!!!")
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;

        # save the reward model in its current state
        self.reward_model.save(bolt.ARTIFACT_DIR, env_id=self.cfg.env, step=self.step)

        print("Reward function is updated!! ACC: " + str(total_acc))

    def save_preference_dataset(self):
        pref_dataset_out_dir = Path(bolt.ARTIFACT_DIR) / f"{self.cfg.env}_preference_dataset_{self.step}"
        pref_dataset_out_dir.mkdir(parents=True, exist_ok=True)
        print(pref_dataset_out_dir, pref_dataset_out_dir.exists())
        self.reward_model.preference_dataset.save(pref_dataset_out_dir, env_id=self.cfg.env, step=self.step)
        print(f"Preference dataset saved to {pref_dataset_out_dir}")

    def save_replay_buffer(self):
        replay_out_dir = Path(bolt.ARTIFACT_DIR) / f"{self.cfg.env}_replay_buffer_{self.step}"
        replay_out_dir.mkdir(parents=True, exist_ok=True)
        print(replay_out_dir, replay_out_dir.exists())
        self.replay_buffer.save(out_directory=replay_out_dir, env_id=self.cfg.env, step=self.step)
        print(f"Replay buffer saved to {replay_out_dir}")

    def save_everything(self):
        self.agent.save(bolt.ARTIFACT_DIR, env_id=self.cfg.env, step=self.step)
        self.reward_model.save(bolt.ARTIFACT_DIR, env_id=self.cfg.env, step=self.step)
        print("Agent and reward models saved to: ", bolt.ARTIFACT_DIR)
        # save anything in the log that has yet to be saved
        self.logger.dump(self.step, save=True)

        # save the preference dataset and the replay buffer
        self.save_preference_dataset()
        self.save_replay_buffer()

    def run(self):
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
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs = self.env.reset()
                if self.cfg.reward_from_image_observations or self.cfg.save_image_observations:
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
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                print("----------------------------------------------")
                print("----------------------------------------------")
                print("----------------------------------------------")
                print(f"Updating the reward model for the first time: step = {self.step}")
                print("----------------------------------------------")
                print("----------------------------------------------")
                print("----------------------------------------------")
                # update schedule
                # TODO: figure out how to change the batch size in the data loader. It looks like reward_schedule is
                #  by default 0, so the batch size does not change. Means we can probably remove this part of model
                #  training
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)

                # save the preference dataset and replay buffer after the update
                # self.save_preference_dataset()
                # self.save_replay_buffer()
                
                # reset Q due to unsupervised exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                        # save the preference dataset and replay buffer after the update
                        pref_dataset_out_dir = Path(bolt.ARTIFACT_DIR) / "preference_dataset"
                        if not pref_dataset_out_dir.exists():
                            os.makedirs(pref_dataset_out_dir)
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)

            for action_rep in range(self.cfg.action_repeat):
                next_obs, reward, done, extra = self.env.step(action)
            # render the current environment if the reward function is using image observations or we are collecting the
            # image-based observations so save
            if self.cfg.reward_from_image_observations or self.cfg.save_image_observations:
                img_next_obs = self._render_image_observation()
            if self.cfg.reward_from_image_observations:
                reward_hat = self.reward_model.r_hat(self.reward_model.format_state_action(img_obs, action))
            else:
                reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the replay buffer and reward training data
            if self.cfg.reward_from_image_observations or self.cfg.save_image_observations:
                self.replay_buffer.add(
                    obs, action, reward_hat,
                    next_obs, done, done_no_max, env_reward=reward,
                    image_observation=img_obs, image_next_observation=img_next_obs)
            else:
                self.replay_buffer.add(
                    obs, action, reward_hat, next_obs, done, done_no_max, env_reward=reward)

            obs = next_obs
            # render the current environment if the reward function is using image observations
            if self.cfg.reward_from_image_observations or self.cfg.save_image_observations:
                img_obs = np.copy(img_next_obs)
            episode_step += 1
            self.step += 1
            interact_count += 1
            
        # save everything involved in this experimental run
        self.save_everything()


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    print("building workspace")
    workspace = Workspace(cfg)
    print("running workspace")
    workspace.run()


if __name__ == '__main__':
    main()
