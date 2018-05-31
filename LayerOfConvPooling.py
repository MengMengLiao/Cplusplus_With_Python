import numpy as np
import theano as th
from theano.tensor.nnet import conv2d
import theano.tensor as T
from theano.tensor.signal import pool
class LayerOfConvPooling(object):

    def __init__(self, rng, Input, filter_shape, image_shape, poolsize=(3, 3)):

        '''
        :param rng: a random number generator used to initialize weights
        :param Input: symbolic image tensor, of shape image_shape
        :param filter_shape: (number of filters, number input feature maps, filter height, filter width)
        :param image_shape:  (batch size, number input feature maps, image height, image width)
        :param poolsize:  the downsampling (pooling) factor (#rows, #cols)
        '''

        assert image_shape[1] == filter_shape[1]

        fan_in  = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W  = th.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=th.config.floatX
            ),
            borrow=True,
            name='w'
        )

        b_values = np.zeros((filter_shape[0],), dtype=th.config.floatX)
        self.b   = th.shared(value=b_values, borrow=True, name='b')

        conv_out   = conv2d(input=Input, filters=self.W, filter_shape=filter_shape, input_shape=image_shape, border_mode=2)

        pooled_out = pool.pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

        self.out    = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.Input  = Input