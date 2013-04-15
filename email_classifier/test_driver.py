import sys
from experiment import Experiment
from feature_parser import FeatureParser

def main():
    try:
        folder_name = sys.argv[1]
        experiment = Experiment(folder_name)
    except IndexError:
        experiment = Experiment()
    experiment.retrieve_data()
    parser = FeatureParser()
    parser.initialize_matrix(experiment.raw_data_set)
    parser.print_global_names()

main()
