#!/bin/bash

script=`readlink -f $0`
basedir=`dirname $script`

echo "Parsing out data for reports from log files..."
$basedir/generate-stats.sh $@

echo "Generating aggregated data for reports..."
$basedir/generate-avg-stats.sh $@

echo "Rendering graphs for test runs..."
$basedir/generate-graphs.sh $@

echo "Rendering graphs for aggregated reports..."
$basedir/generate-avg-graphs.sh $@
