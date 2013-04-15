#!/bin/sh
#
# This is Joel's testing rig because he's lazy. If you want to run a unit test,
# make sure you're in the cs1573 folder and run a command like so (no .py):
# python -m email_classifier.test.test_file

python2 -m email_classifier.test.test_$1
