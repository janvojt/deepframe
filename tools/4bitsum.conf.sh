#!/bin/bash
#
# This is a configuration file for automated launching of a set of tests testing
# different network architectures and learner configurations
# on the 4-bit sum dataset.


# Title will be included in graphs
PROBLEM_TITLE="4-bit sum"

# Network architecture configuration
HIDDEN_NEURONS="10 20 50 100 200 300 400 500 600 8 8,8 8,8,8 8,8,8,8 8,8,8,8,8 8,8,8,8,8,8 8,8,8,8,8,8,8 8,8,8,8,8,8,8,8 8,8,8,8,8,8,8,8,8 8,8,8,8,8,8,8,8,8,8"
INPUT_NEURONS="8"
OUTPUT_NEURONS="5"

# Specify dataset locations within resources
DATASET_LABELS="resources/4bitsum-labels.dat"
DATASET_TESTS="resources/4bitsum-test.dat"

# Learner configuration
EPOCHS="1000"
ITERATIONS="100"
INIT_INTERVALS="-1,1"
LEARNING_RATE="1"
GPU_FLAG="-p"
ADD_OPTS=""

# Testing configuration
MEASURES="mse time memory"
TESTDIR="test-output/4bitsum/m$EPOCHS-l$LEARNING_RATE-a$INIT_INTERVALS$GPU_FLAG"
