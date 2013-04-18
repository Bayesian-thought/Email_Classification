import sys
import re
from experiment import Experiment
from email import Email

def main():
    data = Experiment()
    data.retrieve_data()
    emails = []
    for example in data.raw_data_set:
        emails.append(Email(example[1]))

    i = 1
    with open("dates.csv", "w") as dates_file:
        for email in emails:
            dates_file.write(str(i) + ","+str(email.get_hour()) + "\n")

        
    
main()
