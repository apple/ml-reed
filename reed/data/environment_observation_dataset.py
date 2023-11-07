#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import torch
import typing as t

from BPref.replay_buffer import TrajectoryReplayBuffer
from pathlib import Path
from reed.data.environment_transition_dataset import EnvironmentContrastiveDatapoint, EnvironmentTransitionDataset, \
    EnvironmentContrastiveBatch
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, \
    Grayscale, RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop


JITTER_FACTORS = {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}
DEFAULT_ARGS = {
    'normalization_mean': [0.485, 0.456, 0.406],
    'normalization_std': [0.229, 0.224, 0.225],
    'blur_sigma_min': 0.1,
    'blur_sigma_max': 2.0,
    'jitter_default': 0.,
    'strong_jitter_pval': 0.05,
    'strong_blur_pval': 0.01,
    'strong_crop_scale_min': 0.2,
    'strong_crop_scale_max': 0.7,
    'strong_crop_ratio_min': 1.2,
    'strong_crop_ratio_max': 1.8,
    'weak_jitter_pval': 0.1,
    'weak_blur_pval': 0.,
    'weak_crop_scale_min': 0.8,
    'weak_crop_scale_max': 1.0,
    'weak_crop_ratio_min': 1.6,
    'weak_crop_ratio_max': 1.8,
    'gaussian_blur_kernel_size': 5,
}


class AugmentedEnvironmentObservationDataset(EnvironmentTransitionDataset):
    """
    A dataset of environment observations where the observations are inputs
    and the augmented observations are the target values.

    The dataset can be loaded from a file saved to disk or from a Replay Buffer.

    Observations must be images.
    """

    def __init__(self, replay_buffer: t.Optional[TrajectoryReplayBuffer] = None,
                 file_path: t.Optional[Path] = None,
                 use_strong_augs: bool = False,
                 data_augs: dict = None,
                 height: int = 50,
                 width: int = 50,
                 grayscale_all_images: bool = False,
                 device: str = "cuda",
                 multi_gpu: bool = False,
                 image_formatter: t.Optional[t.Any] = None):
        """
        Either the replay_buffer or the file_path needs to not be of type None. If neither are of type
        None then both are used to populate the dataset

        Args:
            replay_buffer: the buffer of collected environment transitions
            file_path: the location of the datasets file
            use_strong_augs: (default = False) whether to use the stronger augmentation settings.
            data_augs: (default = None) specify how to crop/translate/jitter the data _after_ each image is cropped
                into the same sizes.
                e.g. {
                    'rand_trans': 0.1,      # proportionally shift the image by 0.1 times its height/width
                    'jitter': 0.5,          # probability _factor_ to jitter each of the four jitter factors
                    'grayscale': 0.2,       # got recommended 0.2 or 0.1
                    }
            height， width: (default = 50) crop params can be different but keep final image sizes the same
            grayscale_all_images: (default = False) whether to grayscale the image observations. Note if this is True,
                        no additional grayscale augmentation will be applied.
            device: (default = cuda) whether to run on the cpu or a cuda device
            multi_gpu: (default = False) whether the model is trained across multiple GPUs in which case we do not
                       push the data to a device before returning it
            image_formatter: (default = None) a function to apply to the raw images in order to format them for
                        training.
        """
        assert replay_buffer is not None or file_path is not None, ("One of replay_buffer or file_path must be "
                                                                    "specified. Both are None.")

        assert data_augs, "Must give some basic data-augmentation parameters"

        super(AugmentedEnvironmentObservationDataset, self).__init__(replay_buffer=replay_buffer,
                                                                     file_path=file_path,
                                                                     device=device,
                                                                     multi_gpu=multi_gpu,
                                                                     image_observations=True,
                                                                     image_formatter=image_formatter)
        self.grayscale_all_images = grayscale_all_images
        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.height = height
        self.width = width

        self.transforms = transforms.Compose(self._create_augmentation_transformations())
        print(f"Transformations used: {self.transforms}")

    def _create_augmentation_transformations(self):
        """
        Creates data transformations using the defaults from the Mosaic paper:
            https://github.com/rll-research/mosaic/blob/c033298033ecbd703bb618d1f28f03fdd9643597/mosaic/datasets/multi_task_datasets.py#L162.
        """

        data_augs = self.data_augs
        print("Using strong augmentations?", self.use_strong_augs)
        # Create jitter Transformation.
        jitter_default = DEFAULT_ARGS['jitter_default']
        jitter_val = data_augs.get('strong_jitter', jitter_default) if self.use_strong_augs else data_augs.get('weak_jitter', jitter_default)
        jitters = {k: v * jitter_val for k, v in JITTER_FACTORS.items()}
        jitter = ColorJitter(**jitters)
        jitter_pval = DEFAULT_ARGS['strong_jitter_pval'] if self.use_strong_augs else DEFAULT_ARGS['weak_jitter_pval']
        transformations = [ToTensor(), RandomApply([jitter], p=jitter_pval)]

        if self.use_strong_augs:  # Using "strong" augmentations.
            blur_pval = DEFAULT_ARGS['strong_blur_pval']
            crop_scale = (data_augs.get("strong_crop_scale_min", DEFAULT_ARGS['strong_crop_scale_min']),
                          data_augs.get("strong_crop_scale_max", DEFAULT_ARGS['strong_crop_scale_max']))
            crop_ratio = (data_augs.get("strong_crop_ratio_min", DEFAULT_ARGS['strong_crop_ratio_min']),
                          data_augs.get("strong_crop_ratio_max", DEFAULT_ARGS['strong_crop_ratio_max']))

            if not self.grayscale_all_images:
                # Only add random grayscale if grayscale is not already enabled.
                transformations.append(RandomGrayscale(p=data_augs.get("grayscale", 0)))
            transformations.append(RandomHorizontalFlip(p=data_augs.get('flip', 0)))

        else:  # Using "weak" augmentations.
            blur_pval = DEFAULT_ARGS['weak_blur_pval']
            crop_scale = (data_augs.get("weak_crop_scale_min", DEFAULT_ARGS['weak_crop_scale_min']),
                          data_augs.get("weak_crop_scale_max", DEFAULT_ARGS['weak_crop_scale_max']))
            crop_ratio = (data_augs.get("weak_crop_ratio_min", DEFAULT_ARGS['weak_crop_ratio_min']),
                          data_augs.get("weak_crop_ratio_max", DEFAULT_ARGS['weak_crop_ratio_max']))

        # Add blur, resize crop, and normalization.
        normalization = Normalize(mean=DEFAULT_ARGS['normalization_mean'], std=DEFAULT_ARGS['normalization_std'])
        blur_sigma = (data_augs.get("blur_sigma_min", DEFAULT_ARGS['blur_sigma_min']),
                      data_augs.get("blur_sigma_max", DEFAULT_ARGS['blur_sigma_max']))
        transformations.extend(
            [
                RandomApply([GaussianBlur(kernel_size=DEFAULT_ARGS['gaussian_blur_kernel_size'], sigma=blur_sigma)],
                            p=blur_pval),
                RandomResizedCrop(size=(self.height, self.width), scale=crop_scale, ratio=crop_ratio),
                normalization
            ]
        )

        if self.grayscale_all_images:
            # Apply grayscale to all images.
            transformations.append(Grayscale(1))

        return transformations

    def __getitem__(self, indx: int) -> EnvironmentContrastiveDatapoint:
        """
        Return the specified sample from the dataset
        Args:
            indx: the index of inputs-target pair to be returned

        Returns:
            the environment transition inputs and the target bundled into a single
            datapoint object
        """
        # grab the transition at the given index from the replay buffer
        _, _, _, _, _, _, _, image_observation, _ = self._replay_buffer[indx]

        augmented_observation = self.transforms(image_observation)
        if self.grayscale_all_images:
            grayscale_transform = transforms.Compose([Grayscale(1)])
            image_observation = grayscale_transform(image_observation)

        assert augmented_observation.shape == image_observation.shape, \
            "Augmentations should not alter the original " \
            f"shape. Augmentation shape = {augmented_observation.shape}" \
            f"and image observation shape = {image_observation.shape}"

        return EnvironmentContrastiveDatapoint(state=image_observation.float().to(self._device),
                                               action=None,
                                               augmented_state=augmented_observation.float().to(self._device))

    @staticmethod
    def collate(batch: t.List[EnvironmentContrastiveDatapoint]) -> EnvironmentContrastiveBatch:
        """
        Collate a batch of environment transitions into a batch of environment transitions
        Args:
            batch: a list of environment transition datasets

        Returns:
            a batch of environment transitions
        """
        # used to accumulate the network inputs and targets
        states = []
        next_states = []

        # accumulate inputs and targets from each sample in the batch
        for sample in batch:
            states.append(sample.state)
            next_states.append(sample.augmented_state)

        # bundle the batch of inputs and the batch of targets into a single batch object
        # get item should already have put the tensor on the correct device
        return EnvironmentContrastiveBatch(states=torch.stack(states, dim=0),
                                           actions=None,
                                           augmented_states=torch.stack(next_states, dim=0))

    @property
    def observation_shape(self) -> t.Union[int, t.Sequence[int]]:
        sample_observation = self._replay_buffer.trajectories[0].initial_image_observations
        if self._image_formatter is not None:
            sample_observation = self._image_formatter(sample_observation, batch_states=True)

        return sample_observation.shape[1:]
