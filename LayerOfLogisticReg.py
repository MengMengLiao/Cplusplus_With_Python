import theano as th
import numpy as np
import theano.tensor as T

class LogisticRegression(object):

    def __init__(self, rng, Input, n_in, n_out):

        '''
        :param Input: symbolic variable that describes the input of the architecture
        :param n_in: number of input units, the dimension of the space in which the datapoints lie
        :param n_out: number of output units, the dimension of the space in which the labels lie
        '''

        self.W = th.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=th.config.floatX
            ),
            name='w',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = th.shared(
            value=np.zeros(
                (n_out,),
                dtype=th.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.sigmoid(T.dot(Input, self.W) + self.b)

        self.params = [self.W, self.b]

        self.Input  = Input

    def Cross_entropyErrorFunction(self, y):

        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))