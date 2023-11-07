# Sample-Efficient Preference-based Reinforcement Learning with Dynamics Aware Rewards

This software project accompanies the research paper, [Sample-Efficient Preference-based Reinforcement Learning with Dynamics Aware Rewards](https://openreview.net/pdf?id=i84V7i6KEMd).

This repo forks and builds off of the [BPref](https://github.com/rll-research/BPref) repo.

To run the [SURF](https://openreview.net/pdf?id=TfhfZLQ2EJO), [RUNE](https://arxiv.org/pdf/2205.12401.pdf), and [MetaReward Net](https://openreview.net/pdf?id=OZKBReUF-wX) baselines we compare against in [Paper title](), please use the following repositories. 

- RUNE: https://github.com/rll-research/rune.git
- SURF and MetaReward Net: https://github.com/RyanLiu112/MRN.git

If you find our paper or code insightful, feel free to cite us with the following bibtex:

@inproceedings{metcalf23reed,
  title = {Sample-Efficient Preference-based Reinforcement Learning with Dynamics Aware Rewards},
  author = {Metcalf, Katherine and Sarabia, Miguel and Mackraz, Natalie and Theobald Barry-John},
  booktitle={Conference on Robot Learning},
  year = {2023},
  organization={PMLR},
  url = {https://openreview.net/pdf?id=i84V7i6KEMd}
}

## Documentation

## Getting Started 

To install REED you first need to clone our repository and `cd` into it:

```bash
git clone https://github.com/apple/ml-reed.git
cd ml-REED
```

Then create and run the docker image in `docker/Dockerfile`:

```bash
# Create the docker
cd docker
docker build -t reed --platform linux/amd64 .
# Run the docker
docker run -it --rm reed
```

The docker has a `venv` at `/opt/venv` where most project requirements are already installed. The reed project is 
installed into the docker's `venv`

In the docker image install the project and start the `venv`:

```bash
bash setup.sh
source /opt/venv/bin/activate
```


## Running PEBBLE baselines and REED

All experiments are run through the `reed/experiments/run_preference_experiment.py` script, which takes the following command line arguments:

- `--algorithm`: The algorithm to execute. Must be one of pebble, pebble_image_augmentations, contrastive_reed, or distillation_reed.
- `--task`: The environment and task on which to evaluation the given algorithm. Options are walker_walk, quadruped_walk, and cheetah_run from the DMC Suite and button_press, sweep_into, drawer_open, drawer_close, window_open, and door_open from MetaWorld.
- `--reward_from_images`: Whether to learn the reward using image observations.
- `--preference_labeller`: The BPref synthetic teacher to provide preference labels. Must be one of: equal, mistake, myopic, noisy, oracle, or skip.
- `--trajectory_pair_selection`: The method by which trajectory pairs are selected for preference labelling. 0 is uniform sampling and 1 is disagreement sampling.
- `--max_feedback`: The maximum about of trajectory pairs to be sent for labelling.
- `--out_dir`: The location where results and models should be written.

For example, to run PEBBLE on walker-walk with the oracle labeller, disagreement sampling, 500 pieces of feedback, and joint observations use:

```bash
python reed/experiments/run_preference_experiment.py \
--algorithm pebble \
--task walker_walk \
--preference_labeller oracle \
--trajectory_pair_selection 1 \
--max_feedback 500 \
--out_dir <results/model directory>
```

and to run with image observation add the `--reward_from_images` flag: 

```bash
python reed/experiments/run_preference_experiment.py \
--algorithm pebble \
--task walker_walk \
--preference_labeller oracle \
--trajectory_pair_selection 1 \
--max_feedback 500 \
--reward_from_images \
--out_dir <results/model directory>
```
