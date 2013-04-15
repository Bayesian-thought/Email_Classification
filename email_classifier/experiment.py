import random
import sys
import os
import re

class Experiment(object):
    """
    This is the class that will set up the experiment
        :- Read in data from enron folders
        :- K-fold it
        :- Pass it to data initializer to set up feature matrix
        :- Pass feature matrix to various testing alogirthms
        :- Conduct statistical tests on the different algorithms used
    """

    def __init__(self, folder_name = 'beck'):
        """
        Given folder_name, supplies class level var data_path, which contains the relative path
        to that users email folders.
        Wanted to do fuzzy matching because Rishi is forgetful
        """
        self.data_path = ''
        self.raw_data_set = []

        reg_obj = re.compile(".*" + folder_name + ".*", re.IGNORECASE)
        if reg_obj.search("beck-s"):
            self.data_path = "data/beck-s/"
        elif reg_obj.search("former-d"):
            self.data_path = "data/former-d/"
        elif reg_obj.search("kaminski-v"):
            self.data_path = "data/kaminski-v/"
        elif reg_obj.search("kitchen-l"):
            self.data_path = "data/kitchen-l/"
        elif reg_obj.search("lokay-m"):
            self.data_path = "data/lokay-m/"
        elif reg_obj.search("sanders-r"):
            self.data_path = "data/sanders-r/"
        elif reg_obj.search("williams-w3"):
            self.data_path = "data/williams-w3/"

    def retrieve_data(self):
        """
        Fills list raw_data_set
                :- List of lists  --
                :- [[classification1, path1], [classification2, path2], ...    ]
        """
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), self.data_path)):
            for f in files:
                self.raw_data_set.append( [str(os.path.split(root)[1]), str(os.path.join(root,f))] )

    def print_raw_data_set(self):
        if not self.raw_data_set:
            print "Data initialization error"
            return
        for training_example in self.raw_data_set:
            print training_example[0] + "\t" + training_example[1]

    def k_fold(k=10, randomize=True):
        """
        Shuffles the dataset, then cuts it in to k datasets
        RETURNS:
                List of k shuffled datasets
        """
        pass

    
