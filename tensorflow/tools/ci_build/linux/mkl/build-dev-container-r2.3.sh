#!/usr/bin/env bash

export BAZEL_VERSION=3.1.0
export BUILD_AVX2_CONTAINERS=no
export BUILD_AVX_CONTAINERS=no
export BUILD_CLX_CONTAINERS=no
export BUILD_PY2_CONTAINERS=no
export BUILD_SKX_CONTAINERS=yes
export BUILD_TF_BFLOAT16_CONTAINERS=no
export BUILD_TF_V2_CONTAINERS=yes
export ENABLE_DNNL1=no
export ENABLE_HOROVOD=no
export ENABLE_SECURE_BUILD=yes
export FINAL_IMAGE_NAME=amr-registry.caas.intel.com/aipg-tf/intel-optimized-tensorflow
export HOROVOD_VERSION=
export OPENMPI_DOWNLOAD_URL=
export OPENMPI_VERSION=
export ROOT_CONTAINER=tensorflow/tensorflow
export ROOT_CONTAINER_TAG=2.3.4  # devel
export TF_BUILD_VERSION=r2.3-synced
export TF_BUILD_VERSION_IS_PR=no
export TF_DOCKER_BUILD_DEVEL_BRANCH=r2.3-synced
export TF_DOCKER_BUILD_VERSION=r2.3-synced
export TF_REPO=https://github.com/Intel-tensorflow/tensorflow

bash ./build-dev-container.sh
