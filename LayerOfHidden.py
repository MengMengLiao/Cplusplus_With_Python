import numpy as np
import theano as th
from theano.tensor.nnet import relu
import theano.tensor as T
class HiddenLayer(object):

    def __init__(self, rng, Input, n_in, n_out, W=None, b=None, activation=T.tanh):

        '''
        :param rng: a random number generator used to initialize weights
        :param Input: a symbolic tensor of shape (n_examples, n_in)
        :param n_in:  dimensionality of input
        :param n_out: number of hidden units
        :param activation: Non linearity to be applied in the hidden layer
        '''

        self.Input = Input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=th.config.floatX
            )
            if activation == th.tensor.nnet.sigmoid:
                W_values *= 4

            W = th.shared(value=W_values, name='w', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=th.config.floatX)
            b = th.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output  = T.dot(self.Input, self.W) + self.b

        self.output = activation(lin_output)

        self.params = [self.W, self.b]