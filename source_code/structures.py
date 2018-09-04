"""This file stores the data structures we used.

The DataSet class is a wrapper for the ([values...], class) format our
training data is in. It also contains some metadata that keeps track
of all the attributes and their valid values. It also implements
__len__ and __next__ so we can use it like a normal collection.

The DistanceCounter class consumes the K nearest neighbors of a tuple
processed through KNN and attempts to classify it, using weighted
voting from each neighboring point.
"""

from copy import deepcopy
from math import sqrt, exp

class DataSet:
    """This will organize tuples in such a way that when calculating the
    information gained from splitting on a certain attribute, we can tell
    the algorithm to simply ignore the weight of a certain attribute
    without having to perform a deep copy of the training data during the
    actual splitting of the data set.

    Implements __len__ and __iter__, so it can be used in a for/in
    loop.

    """
    
    def __init__(self):
        self.tuples = []
        self.classes = []

        
    def __len__(self):
        return len(self.tuples)

    
    def __iter__(self):
        return self.tuples.__iter__()

    
    def __getitem__(self, key):
        return self.tuples[key]


    def _euclidian_distance(point1, point2):
        pre_root_sum = 0
        idx = 0
        while idx < len(point1):
            pre_root_sum += (point1[idx] - point2[idx])**2
            idx += 2
        return sqrt(pre_root_sum)
    
    
    def addTuple(self, tup):
        self.tuples.append(tup)

        
    def pointAt(self, loc):
        """Returns just the point-part, excluding the class"""
        return self.tuples[loc][0]
        
    
    def distanceTo(self, loc, second_point):
        return DataSet._euclidian_distance(self.pointAt(loc),
                                           second_point)

    
    def getDistances(self, second_point):
        distances = []
        for index, tr_sample in enumerate(self.tuples):
            classifier = tr_sample[1]
            dist = self.distanceTo(index, second_point)
            distances.append((classifier, dist))
        return distances

    
    def from_tuple_list(tuples):
        """ Create a DataSet from the formatted training data."""
        new_set = DataSet()
        
        for tup in tuples:
            if tup[1] not in new_set.classes:
                new_set.classes.append(tup[1])
            new_set.addTuple(tup)

        return new_set        

    
    def enumerate_classes(self):
        """Need to convert the expected values in a form we can compare
        to the outputs of the network; each class is enumerated by
        'i' and is represented by a list where l[i] = 1, all other
        values 0"""
        
        classifiers = {}
        for index, c in enumerate(self.classes):
            class_representation = [0]*len(self.classes)
            class_representation[index] = 1
            classifiers[c] = class_representation

        return classifiers

    def truth_vectors(self):
        expected = [None]*len(self)
        classifiers = self.enumerate_classes()
        for index in range(0, len(self)):
            expected[index] = classifiers[self[index][1]]
        return expected

    
    def normalize(self):
        """Compress the values of each attribute into a range between -1 and 1
        using the maximum and minimum for each attribute."""
        for index in range(0, len(self[0][0]) - 1):
            max_val = max([tup[0][index] for tup in self])
            min_val = min([tup[0][index] for tup in self])
            extent = max(abs(max_val), abs(min_val))
            if extent == 0:
                continue
            for tup in self:
                tup[0][index] /= extent
    
class DistanceCounter:
    """This class will consume the list of K-nearest neighbors found in
    the KNN algorithm and then make a guess of what classification the
    given tuple belongs to."""
    def __init__(self, data_set, neighbors):
        """Counts up how many neighbors are in each classification class."""
        self.counts = {}
        for classification in data_set.classes:
            self.counts.update({classification: [0, 0]})
        # Get weighted votes
        for classifier, dist in neighbors:
            self.counts[classifier][0] += 1
            self.counts[classifier][1] += dist

    def getNearestClass(self):
        """Classify a tuple based on the *medoid* of each classification
        class, with the similarity measure 1/(total_distance * (number
        in class ** 2))"""
        
        nearest_so_far = 100000000
        nearest_neighbor = None
        for classifier, similarity in self.counts.items():
            if similarity[0] == 0:
                continue
            else:
                smidge = 1 + exp(similarity[0])
                closeness = 1/(similarity[1]*similarity[0]*smidge)
                if closeness < nearest_so_far:
                    nearest_so_far = closeness
                    nearest_neighbor = classifier
                
        return nearest_neighbor
    
