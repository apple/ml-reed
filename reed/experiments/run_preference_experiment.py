#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

"""Submit many children jobs."""
import os
from pathlib import Path
from argparse import ArgumentParser

import yaml


SEEDS = [12345, 23451, 34512, 45123, 51234, 67890, 78906, 89067, 90678, 6789]


LABELLER_TO_CONFIG = {"equal": "reed/experiment_configs/labeller/equal.yaml",
                      "mistake": "reed/experiment_configs/labeller/mistake.yaml",
                      "myopic": "reed/experiment_configs/labeller/myopic.yaml",
                      "noisy": "reed/experiment_configs/labeller/noisy.yaml",
                      "oracle": "reed/experiment_configs/labeller/oracle.yaml",
                      "skip": "reed/experiment_configs/labeller/skip.yaml"}

TASK_TO_CONFIG = {
    "walker_walk": "reed/experiment_configs/tasks/walker_walk.yaml",
    "quadruped_walk": "reed/experiment_configs/tasks/quadruped_walk.yaml",
    "cheetah_run": "reed/experiment_configs/tasks/cheetah_run.yaml",
    "button_press": "reed/experiment_configs/tasks/button_press.yaml",
    "sweep_into": "reed/experiment_configs/tasks/sweep_into.yaml",
    "drawer_open": "reed/experiment_configs/tasks/drawer_open.yaml",
    "drawer_close": "reed/experiment_configs/tasks/drawer_close.yaml",
    "window_open": "reed/experiment_configs/tasks/window_open.yaml",
    "door_open": "reed/experiment_configs/tasks/door_open.yaml"
}


def run_experiment(
        algorithm: str,
        reward_from_images: bool,
        preference_labeller: str,
        trajectory_pair_selection: int,
        task: str,
        max_feedback: int,
        out_dir: str
) -> None:
    """
    Run the specified PbRL experiment
    Args:
        algorithm: the PbRL algorithm. Must be one of PEBBLE or REED.
        reward_from_images: when true the state observations are images and joint positions when false
        preference_labeller: the strategy to providing synthetic preference labels. Must be one of: equal, mistake,
                             myopic, noisy, oracle, or skip.
        trajectory_pair_selection: the strategy for selecting trajectory pairs for preference labelling. Must be one of
                                   0 (uniform) or 1 (disagreement).
        task: The environment and task on which to evaluation the given algorithm. Options are walker_walk,
              quadruped_walk, and cheetah_run from the DMC Suite and button_press, sweep_into, drawer_open,
              drawer_close, window_open, and door_open from MetaWorld.
        max_feedback: the maximum number of trajectory pairs sent to the preference labeller for feedback
        out_dir: the directory where results and models will be written
    """
    # load the config for the task
    with open(TASK_TO_CONFIG[task], 'r') as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        task_config = yaml.load(f, Loader=yaml.FullLoader)

    # the number of updates to make to the reward network depends on the environment (DMC vs MetaWorld)
    if "metaworld" in task_config['env']:
        num_updates = 200
        action_repeat = 5
    else:
        num_updates = 10
        action_repeat = 1

    # get the PEBBLE experiment configuration
    # this is used for both PEBBLE and REED algorithms as REED extends and builds off of PEBBLE
    pebble_config = task_config["pebble"]

    # get the REED configuration
    reed_config = task_config[algorithm] if "reed" in algorithm else None

    # load the labeller configuration
    with open(LABELLER_TO_CONFIG[preference_labeller], 'r') as f:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        labeller_config = yaml.load(f, Loader=yaml.FullLoader)
    # convert the labeller configuration into command line arguments
    labller_cmnd_args = " ".join([f"{param_name}={param_value}"
                                  for param_name, param_value
                                  in labeller_config.items()])

    # the batch size is the same as the number of samples given to the teacher each time
    # teacher feedback is requested
    pebble_config["reward_train_batch"] = int(max_feedback / num_updates)
    pebble_config["preference_dataset_update_size"] = int(max_feedback / num_updates)

    # the script used to run the experiment
    if algorithm == "pebble" and not reward_from_images:
        experiment_script = "reed/experiments/train_PEBBLE.py"
    elif algorithm == "pebble":
        experiment_script = "reed/experiments/train_PEBBLE_images.py"
    elif algorithm == "pebble_image_augmentations":
        experiment_script = "reed/experiments/train_PEBBLE_image_augmentations.py"
    elif algorithm == "reed" and not reward_from_images:
        experiment_script = "reed/experiments/train_REED.py"
    else:
        experiment_script = "reed/experiments/train_REED_images.py"

    # we need to prepend a command to our script
    experiment_command = f"export MUJOCO_GL='egl'; chmod +x {experiment_script};python3 {experiment_script}"

    # convert the pebble configuration to command line arguments
    pebble_cmd_args = " ".join([f"{param_name}={param_value}"
                                for param_name, param_value
                                in pebble_config.items()])

    # build the complete (minus random seed) command line arguments
    if reed_config is not None:
        # convert the experiment configuration to command line arguments
        reed_args = " ".join([f"{param_name}={param_value}"
                             for param_name, param_value in reed_config.items()])

        cmd_args = " ".join([f"env={task_config['env']}", f"feed_type={trajectory_pair_selection}",
                             f"max_feedback={max_feedback}",
                             f"reward_from_image_observations={reward_from_images}",
                             f"action_repeat={action_repeat}",
                             pebble_cmd_args, reed_args, labller_cmnd_args])
    else:
        cmd_args = " ".join([f"env={task_config['env']}", f"feed_type={trajectory_pair_selection}",
                             f"max_feedback={max_feedback}",
                             f"reward_from_image_observations={reward_from_images}",
                             f"action_repeat={action_repeat}",
                             pebble_cmd_args, labller_cmnd_args])

    # need to run the experiment per seed
    for seed in SEEDS:
        # create an out directory per seed
        seed_out_dir = Path(out_dir)
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        # add the random seed to the command line arguments
        arguments_str = f"seed={seed} out_dir={seed_out_dir} {cmd_args}"
        # run the experiment
        print(f"Executing: {experiment_command} {arguments_str}")
        os.system(f"{experiment_command} {arguments_str}")
        import sys; sys.exit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Execute a preference learning experiment with either REED or PEBBLE.")
    parser.add_argument('--algorithm',
                        default="contrastive_reed",
                        type=str,
                        help="The algorithm to execute. Must be one of pebble, pebble_image_augmentations, "
                             "contrastive_reed, or distillation_reed.",
                        choices=["pebble", "pebble_image_augmentations", "contrastive_reed", "distillation_reed"]
                        )
    parser.add_argument('--reward_from_images',
                        action="store_true",
                        help="Whether to learn the reward using image observations."
                        )
    parser.add_argument('--preference_labeller',
                        default="oracle",
                        help="The BPref synthetic teacher to provide preference labels. Must be one of: equal, mistake,"
                             " myopic, noisy, oracle, or skip.",
                        choices=LABELLER_TO_CONFIG.keys()
                        )
    parser.add_argument('--trajectory_pair_selection',
                        default=1,
                        type=int,
                        help="The method by which trajectory pairs are selected for preference labelling. 0 is uniform "
                             "sampling and 1 is disagreement sampling.",
                        choices=[0, 1]
                        )
    parser.add_argument('--task',
                        default="walker_walk",
                        help="The environment and task on which to evaluation the given algorithm. Options are "
                             "walker_walk, quadruped_walk, and cheetah_run from the DMC Suite and button_press, "
                             "sweep_into, drawer_open, drawer_close, window_open, and door_open from MetaWorld.",
                        choices=TASK_TO_CONFIG.keys()
                        )
    parser.add_argument('--max_feedback',
                        default=500,
                        type=int,
                        help="The maximum about of trajectory pairs to be sent for labelling.")
    parser.add_argument('--out_dir',
                        required=True,
                        help="The location where results and models should be written.")
    args = parser.parse_args()

    run_experiment(**vars(args))
