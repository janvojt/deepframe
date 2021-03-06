#!/bin/bash
#
# This is a configuration file for automated launching of a set of tests testing
# different network architectures and learner configurations
# on the MNIST dataset.


# Title will be included in graphs
PROBLEM_TITLE="MNIST"


# Network architecture configuration
## We want to use file configurations for the networks.
CONF_DIR="examples/cnn-mnist-test-lr"

# Specify dataset locations within resources
DATASET_LABELS="resources/mnist/train-images-idx3-ubyte"\
":resources/mnist/train-labels-idx1-ubyte"
DATASET_TESTS="resources/mnist/t10k-images-idx3-ubyte"\
":resources/mnist/t10k-labels-idx1-ubyte"

# Learner configuration
EPOCHS="1"
ITERATIONS="20"
INIT_INTERVALS="-0.3,0.3"
LEARNING_RATE="0.001"
GPU_FLAG="-p"
ADD_OPTS="-i"

# Testing configuration
MEASURES="mse time memory"
TESTDIR="test-output/mnist/cnn/float/m$EPOCHS-lr$GPU_FLAG"
