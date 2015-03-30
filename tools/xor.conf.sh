#!/bin/bash
#
# This is a configuration file for automated launching of a set of tests testing
# different network architectures and learner configurations
# on the XOR binary operator dataset.


# Title will be included in graphs
PROBLEM_TITLE="XOR operator"

# Network architecture configuration
HIDDEN_NEURONS="10 20 50 100 200 300 400 500 600 8 8,8 8,8,8 8,8,8,8 8,8,8,8,8 8,8,8,8,8,8 8,8,8,8,8,8,8 8,8,8,8,8,8,8,8 8,8,8,8,8,8,8,8,8 8,8,8,8,8,8,8,8,8,8"
INPUT_NEURONS="2"
OUTPUT_NEURONS="1"

# Specify dataset locations within resources
DATASET_LABELS="xor"
DATASET_TESTS="xor"

# Learner configuration
EPOCHS="1000"
ITERATIONS="100"
INIT_INTERVALS="-0.3,0.3"
LEARNING_RATE="1"
GPU_FLAG="-p"
ADD_OPTS=""

# Testing configuration
MEASURES="mse time memory"
TESTDIR="test-output/xor/m$EPOCHS-l$LEARNING_RATE-a$INIT_INTERVALS$GPU_FLAG"
