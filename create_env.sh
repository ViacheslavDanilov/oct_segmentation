#!/usr/bin/env bash

# :::::::::::::::::: Options ::::::::::::::::::
PYTHON_VERSION=$( bc <<< "3.8" )
ENV_NAME="oct"
REQUIREMENTS_PATH="requirements/dev.txt"
# :::::::::::::::::::::::::::::::::::::::::::::

conda update -n base -c defaults conda --yes
conda create --name ${ENV_NAME} python=${PYTHON_VERSION} --no-default-packages --yes
conda init --all --dry-run --verbose
conda activate ${ENV_NAME}
python -V
pip install -r ${REQUIREMENTS_PATH} --no-cache-dir
