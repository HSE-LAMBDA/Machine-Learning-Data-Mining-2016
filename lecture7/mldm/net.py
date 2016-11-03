import numpy as np

import theano
import theano.tensor as T

from lasagne import *

class Net(object):
    def save(self, path):
        return np.savez(path, *layers.get_all_param_values(self.net))
    
    def load(self, path):
        with np.load(path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            layers.set_all_param_values(self.net, param_values)

        return self
    
    def __str__(self):
        def get_n_params(l):
            return np.sum([
                np.prod(l.get_params()[0].get_value().shape)
                for param in l.get_params()
            ])

        return '\n'.join([
            '%s\n  output shape:%s\n  number of params: %s' % (l, l.output_shape, get_n_params(l))
            for l in layers.get_all_layers(self.net)
        ])