"""This file allows us to read in data from a CSV and create a DataSet
with it.
"""
from .structures import DataSet
import csv

def data_from_csv(filename, classifier_index, exclude_indices=[]):
    """Creates a DataSet from a CSV file, using the attribute at the
    specified index as the classification class.

    The first row in the CSV file MUST contain the names of each
    attribute; if the CSV data is consistent, this should be the only
    step that must be performed by-hand.

    Assumes the classification class of each tuple is an integer 
    value.
    """

    tuple_list = []
    missing_value_locations = [] #list of (row, attr) pairs
    
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        attribute_names = reader.__next__()
        

        for set_index, row in enumerate(reader):
            #################### error checking
            if len(row) != len(attribute_names):
                print("{}row, {}head"
                      .format(len(row), len(attribute_names)))
                raise IOError("CSV attribute names \
                do not match number of values per tuple.")

            #################### Construct the tuple
            classification_class = None
            new_tup = []
            hit_class = 0
            for index, value in enumerate(row):
                if index == classifier_index:
                    # Assumes classifier is an int
                    classification_class = value
                elif index in exclude_indices:
                    continue
                else:
                    new_tup.append(float(value))

            tuple_list.append((new_tup, classification_class))
            
    return DataSet.from_tuple_list(tuple_list)
