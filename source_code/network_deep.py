from typing import List, Tuple, Callable
from math import exp, inf
from random import random, randint

from parse import data_from_csv
from structures import DataSet

import numpy as np

class GNode:
    """A wrapper for a graph node."""
    def __init__(self, index:int, data):
        self.index:int = index
        self.data = data
        self.w_input = 0
        
    def activate(self, source):
        self.w_input = source
        self.data = 1/(1 + exp(-source))

        
    def setData(self, new_data):
        self.data = new_data

    def setInput(self, inp):
        self.w_input = inp


class Network:
    """Implements a three-layer neural network."""


    def _sigmoid(val:float) -> float:
        """Activation function for each neuron."""
        return 1/(1 + exp(-val))

    def _d_sigmoid(val):
        return Network._sigmoid(val) * (1 - Network._sigmoid(val))

    d_sigmoid_vec = np.vectorize(_d_sigmoid)

    def _initialize_arc() -> float:
        """Sets initial weight of each arc in the network"""
        return random()


    def __init__(self, layer_counts):


        self.layer_counts = layer_counts
        self.depth = len(self.layer_counts)
        
        # learning rate
        self.learning_rate = 0.75

        self.biases = np.array([.5]*self.depth)
        
        # Initialize the arcs. The weight matrix is laid out so that
        # row n is all the inputs into the nth node of the next layer.
        self.weights = [None]*(self.depth-1)
        self.biases = [None]*(self.depth-1)
        for x in range(0, self.depth-1):
            ins, outs = layer_counts[x], layer_counts[x + 1]
            self.weights[x] = np.random.rand(outs, ins)
            self.biases[x] = np.array([.5]*outs)

        self.weights = np.array(self.weights)
            
        self.neurons = [None]*self.depth
        node_icount = 0
        for x in range(0, self.depth):
            self.neurons[x] = np.array([GNode(y + node_icount, 0) for y in range(0, layer_counts[x])])
            node_icount += layer_counts[x]

        # for convenience
        self.inputs = self.neurons[0]
        self.outputs = self.neurons[self.depth - 1]

    def setHiddenBias(self, bias):
        self.hidden_bias = bias


    def setOutputBias(self, bias):
        self.output_bias = bias


    def _readInto(self, input_data: List):
        """Loads a list of numerical inputs into the input neurons."""
        if len(input_data) < len(self.inputs):
            raise ValueError('#input attrs differs from #input nodes.')
        for index, n in enumerate(self.inputs):
            n.setData(input_data[index])


    def _readOut(self, maximal=False) -> List[float]:
        """Returns a list containing the outputs of each output neuron."""

        outputs = np.array([n.data for n in self.outputs])
        if maximal:
            return max(outputs)
        else: return outputs


    def _squash(self, source_layer):
        """transmit the outputs from each node in source into each node in
        sink, and activate the neuron with the given inputs."""
        
        if source_layer > self.depth - 1 or source_layer < 0:
            raise ValueError("Trying to squash from non-existant layer {}"
                             .format(source_layer))
        source_vector = np.array([n.data for n in self.neurons[source_layer]])
        for index, neuron in enumerate(self.neurons[source_layer + 1]):
            neuron.activate(source_vector.dot(self.weights[source_layer][index])
                            + self.biases[source_layer][index])
        

    def enum_arcs(self):

        arcs = {}
        start_ind = 0
        for ind, lay in enumerate(self.weights):
            transp = lay.T
            from_len = len(transp)
            out_start = from_len + start_ind
            for in_ind, row in enumerate(transp):
                arcs.update({in_ind + start_ind: {}})
                for out_ind, arcval in enumerate(row):
                    arcs[in_ind + start_ind].update({out_ind + out_start: arcval})
            start_ind += from_len            
        return arcs

    
    def propagate(self):
        """sends inputs forward through NN."""
        for x in range(0, self.depth - 1):
            self._squash(x)
        return self._readOut()

    
    def backprop(self, input_values, expected_output):

        self._readInto(input_values)
        outputs = self.propagate()

        # Compute the error in this layer
        pre_activate = np.array([n.w_input for n in self.outputs])
        sigma_prime = self.d_sigmoid_vec(pre_activate)
        error_output = (outputs - expected_output) * sigma_prime
        
        next_layer_error = error_output.reshape(-1, 1)
        prev_outs = np.array([n.data for n in self.neurons[-2]]).reshape(1, -1)
        
        # make adjustments to output layer
        self.weights[-1] -= ((next_layer_error @ prev_outs) * self.learning_rate)
        self.biases[-1] -= (next_layer_error.reshape(1, -1)[0] * self.learning_rate)        
        for layer_index in range(2, self.depth):
            prev_outs = np.array([n.data for n in self.neurons[-layer_index-1]]).reshape(1, -1)
            sigma_prime = self.d_sigmoid_vec([n.w_input for n in self.neurons[-layer_index]])
            w_t = self.weights[-layer_index + 1].T


            next_layer_error = (w_t @ next_layer_error) * sigma_prime.reshape(-1, 1)
            
            # adjust internal weights
            self.weights[-layer_index] -= next_layer_error @ prev_outs * (self.learning_rate/layer_index)
            self.biases[-layer_index] -= (next_layer_error.reshape(1, -1)[0] * self.learning_rate/layer_index)

        #return output error to be used for training
        return outputs
    
    def classify(self, input_point):
        """Attempts to classify a list of numerical inputs. The index of the
        neuron with the maximum output value is considered to be the
        classification class."""
        self._readInto(input_point)
        output = self.propagate()
        return np.argmax(output)


    # Train a small amount and return JSON-serializable data
    def train_alittle(self, t_d, batch_size=100, truth_vector=None):
        from random import randint
        randoms = [randint(0, len(t_d)-1) for _ in range(0,batch_size)]
        trainers = [np.array(t_d[rand][0]) for rand in randoms]
        tvs = t_d.truth_vectors()
        expected = [tvs[rand] for rand in randoms]

        square_sum = np.vectorize(lambda expect, output: (output - expect)**(2))

        total_error = 0
        
        for ind, sample in enumerate(trainers):
            prop = self.backprop(sample, expected[ind])
            mse = sum(square_sum(prop, expected[ind])) / len(prop)

            total_error += mse

        arcs = self.enum_arcs()
        data = {'arcs': arcs}
        data['topology'] = self.layer_counts
        data['errRate'] =  total_error / batch_size
        return data
    

    def train(self, t_d, truth_vector=None):
        """Repeatedly backpropagates error until mean square error is
        minimized. Stops training if total error reduction does not
        decrease significantly."""
        pts = [np.array(pair[0]) for pair in t_d]

        if truth_vector is None:
            expected = t_d.truth_vectors()
        else: expected = [truth_vector]
        
        propagations = 0
        total_error = 0
        previous_error = inf

        from random import randint

        square_sum = np.vectorize(lambda expect, output: (output - expect)**(2))
        
        max_tests = 1000000

        for x in range(0, max_tests):
            ran = randint(0, len(pts)-1)

            # adjust weights and read out
            prop = self.backprop(pts[ran], expected[ran])
            mse = sum(square_sum(prop, expected[ran])) / len(prop)
            
            total_error += mse
            propagations += 1
            
            # Check if error is actually being reduced
            if x % (1000) == 0:
                current_error = total_error / propagations

                # End the training if the change in error every 1000
                # does not change significantly
                if previous_error - current_error < 0.001:
                    break
                
                previous_error = current_error

                print("err {:.3f}".format(current_error))
                
    def test(self, training_data, tests):
        """Check network's accuracy by running a set number of tests against
        training data."""
        classifiers = training_data.enumerate_classes()
        correct = 0
        print("#"*30, "Testing", "#"*30)
        for x in range(0, tests):
            pair = training_data[x]
            expected = int(pair[1])
            classifier = self.classify(pair[0])
            if classifier == expected:
                correct += 1
            if x % (tests/10) == 0:
                print(self.propagate())
                print("{}, {}".format(classifier, expected))        
        return correct / float(tests)
    
def main():
    t_d = data_from_csv("data_banknote_authentication.csv", 4)
    t_d.normalize()
    
    randoms = [randint(0, len(t_d) - 10) for _ in range(0, 10)]
    tups = []
    for x in randoms:
        tups.append(t_d.tuples.pop(x))
    test_data = DataSet.from_tuple_list(tups)

    print("#"*30, "Training", "#"*30)
    nn = Network([4, 8, 2])
    nn.train(t_d)
    
    result = nn.test(test_data, 10)

    print("Accuracy rating: {}".format(result))
    
main()
