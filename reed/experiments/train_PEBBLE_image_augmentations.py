#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

#!/usr/bin/env python3

import hydra

from reed.algorithms.pebble_image_augmentations import PEBBLEImageAugmentations


"""
Run the PEBBLE PbRL algorithm with an image augmentation auxiliary task.
"""


@hydra.main(config_path='../experiment_configs/train_PEBBLE_image_augmentations.yaml', strict=True)
def main(cfg):
    workspace = PEBBLEImageAugmentations(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
