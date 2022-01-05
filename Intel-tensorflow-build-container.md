# Steps to generate a container with Intel® Optimization for TensorFlow

This guide will help you generate a container with Intel's r2.8 branch.

## Steps:

1. Clone intel-tensorflow r2.8 branch:

    ```
    $ git clone https://github.com/Intel-tensorflow/tensorflow.git --branch=r2.8 --single-branch
    $ cd tensorflow
    $ git checkout v2.8.0
    # Run "git log" and check for the right git hash
    ```

2.  Go to the directory that has Intel mkl docker files:

    ```
    $ cd tensorflow/tools/ci_build/linux/mkl/
    ```

3.  Run build-dev-container.sh by passing the following env parameters:

    For AVX containers:

    ```
    $ env  ROOT_CONTAINER=tensorflow/tensorflow \
    	ROOT_CONTAINER_TAG=devel \
    	TF_DOCKER_BUILD_DEVEL_BRANCH=v2.8.0 \
    	TF_REPO=https://github.com/Intel-tensorflow/tensorflow \
    	TF_DOCKER_BUILD_VERSION=2.8.0 \
    	BUILD_AVX_CONTAINERS=yes \
    	BUILD_TF_V2_CONTAINERS=yes \    	
    	BAZEL_VERSION=4.2.1 \    	
    	ENABLE_SECURE_BUILD=yes \
            ENABLE_HOROVOD=yes \
    	./build-dev-container.sh > ./container_build.log
    ```

    For AVX512 containers:

    ```
    $ env  ROOT_CONTAINER=tensorflow/tensorflow \
    	ROOT_CONTAINER_TAG=devel \
    	TF_DOCKER_BUILD_DEVEL_BRANCH=v2.8.0 \
    	TF_REPO=https://github.com/Intel-tensorflow/tensorflow \
    	TF_DOCKER_BUILD_VERSION=2.8.0 \
    	BUILD_SKX_CONTAINERS=yes \
    	BUILD_TF_V2_CONTAINERS=yes \    	
    	BAZEL_VERSION=4.2.1 \    	
    	ENABLE_SECURE_BUILD=yes \
            ENABLE_HOROVOD=yes \
    	./build-dev-container.sh > ./container_build.log
    ```  

4.  Open a second terminal session at the same location and run `tail -f container_build.log` to monitor container build progress
    or wait until the build finishes and then open the log file <container_build.log> ...

    ```
    INFO: Build completed successfully, 18811 total actions.
    ```

    Below output indicates that the container has intel-optimized tensorflow:

    ```
    PASS: MKL enabled test in <intermediate container name>
    ```

5.  Check if the image was built successfully and tag it:

    AVX container:

    ```
    $ docker images
    intel-mkl/tensorflow:2.8.0-devel-mkl
    $ docker tag intel-mkl/tensorflow:2.8.0-devel-mkl intel/intel-optimized-tensorflow:2.8.0-devel-mkl
    ```   

    AVX512 container:

    ```
    $ docker images
    intel-mkl/tensorflow:2.8.0-avx512-devel-mkl
    
    $ docker tag intel-mkl/tensorflow:2.8.0-avx512-devel-mkl intel/intel-optimized-tensorflow:2.8.0-avx512-devel-mkl
    ``` 
