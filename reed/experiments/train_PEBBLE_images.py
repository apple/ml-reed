#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

#!/usr/bin/env python3

import hydra

from reed.algorithms.pebble import PEBBLE


"""
Run the PEBBLE PbRL algorithm with images as the state observation.
"""


@hydra.main(config_path='../experiment_configs/train_PEBBLE_images.yaml', strict=True)
def main(cfg):
    workspace = PEBBLE(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
