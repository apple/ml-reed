#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

#!/bin/bash -ue

# Final mujoco_py check
python3.9 -c "import sys; import mujoco_py; sys.exit(mujoco_py.builder.get_nvidia_lib_dir() is None)" || \
  echo "mujoco_py cannot find NVIDIA's EGL implementation"

export FLIT_ROOT_INSTALL=1

python3.9 -m pip install --upgrade pip


echo "-------------- installing mujoco, mujoco dmc2gym, and dm_control ---------------------------------------"
python3.9 -m pip install ./custom_dmc2gym

echo "-------------- flit installing up BPref ---------------------------------------"
python3.9 -m flit -f ./pyproject_bpref.toml install -s

echo "-------------- flit installing REED ---------------------------------------"
python3.9 -m flit -f ./pyproject_reed.toml install -s
