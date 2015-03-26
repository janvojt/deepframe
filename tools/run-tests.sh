#!/bin/bash
#
# This script launches a set of tests testing different network architectures
# and learner configurations.


# Disable limits on process CPU time if any
ulimit -t unlimited

script=`readlink -f $0`
basedir=`dirname $script`

# Process options
USAGE="Usage: $0 configuration-file [network-overrides]"
if [ "$#" == "0" ]; then
	echo "$USAGE"
	exit 1
elif [ ! -f "$1" ]; then
	echo "File '$1' not found."
	echo "$USAGE"
	exit 1
fi

# Read configuration
source "$1"
shift

# Run the tests
TEST_OUT="$TESTDIR"
RESOURCES_DIR="resources"
EXEC="bin/ffwdnet"

cd "$basedir/.."
mkdir -p "$TEST_OUT"

for l in $HIDDEN_NEURONS; do
for (( i=1;i<=$ITERATIONS;i++ )); do
	(/usr/bin/time -f "%U user\n%S system\n%e real\n%M max memory (kB)\n" \
		"$EXEC" \
		-m "$EPOCHS" \
		-l "$LEARNING_RATE" \
		-a "$INIT_INTERVALS" \
		-e -1 \
		-s "$DATASET_LABELS" \
		-t "$DATASET_TESTS" \
		-c "$INPUT_NEURONS,$l,$OUTPUT_NEURONS" \
		"$GPU_FLAG" \
		$ADD_OPTS \
		$@ \
		) &>> "$TEST_OUT/test-$l.log"
done
done
