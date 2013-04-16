#!/bin/sh

java -cp weka-3-7-9/weka.jar:. weka.classifiers.functions.MultilayerPerceptron -t $1 -x 5 -N 250 -H 2,1
