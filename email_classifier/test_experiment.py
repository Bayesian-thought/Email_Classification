import os
import sys
from experiment import Experiment

def main():
    if len(sys.argv) < 2:
            sys.stderr.write("Need to pass user folder name\n")
            return

    folder_name = sys.argv[1]
    experiment = Experiment(folder_name)
	experiment.retrieve_data()
	parser = FeatureParser(experiment.raw_data_set)
	(names, secondary_names, body_text) = parser.get_mail_elements("data/beck-s/moscoso__mike/4.")
	print names
	print secondary_names
	print body_text
	
main()
