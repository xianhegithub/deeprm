import numpy as np
import theano, theano.tensor as T
import lasagne
from collections import OrderedDict



class PGLearner:
    def __init__(self, pa):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        self.num_frames = pa.num_frames

        self.update_counter = 0

        states = T.tensor4('states')  # 4-dimensional array

        print ('network_input_height=', pa.network_input_height)
        print ('network_input_width=', pa.network_input_width)
        print ('network_output_dim=', pa.network_output_dim)

        # image representation
        self.l_out = build_pg_network(pa.network_input_height, pa.network_input_width, pa.network_output_dim)   # build the neural network with different layers, using lasagna. return only the ouput layer.

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        params = lasagne.layers.helper.get_all_params(self.l_out)

        print (' params=', params, ' count=', lasagne.layers.count_params(self.l_out))

        self._get_param = theano.function([], params)   # define a function that outputs all the parameters.

        # ===================================
        # training function part
        # ===================================

        prob_act = lasagne.layers.get_output(self.l_out, states)  # read the output of the network

        # --------  Supervised Learning  --------

        su_target = T.ivector('su_target')


        su_loss = lasagne.objectives.categorical_crossentropy(prob_act, su_target)
        su_loss = su_loss.mean()

        l2_penalty = lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l2)
        # l1_penalty = lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l1)

        su_loss += 1e-3*l2_penalty
        print ('lr_rate=', self.lr_rate)

        su_updates = lasagne.updates.rmsprop(su_loss, params,
                                             self.lr_rate, self.rms_rho, self.rms_eps)
        #su_updates = lasagne.updates.nesterov_momentum(su_loss, params, self.lr_rate)

        self._su_train_fn = theano.function([states, su_target], [su_loss, prob_act], updates=su_updates)

        self._su_loss = theano.function([states, su_target], [su_loss, prob_act])

        self._debug = theano.function([states], [states.flatten(2)])

    #  -------- Supervised Learning --------
    def su_train(self, states, target):
        loss, prob_act = self._su_train_fn(states, target)
        return np.sqrt(loss), prob_act

    def su_test(self, states, target):
        loss, prob_act = self._su_loss(states, target)
        return np.sqrt(loss), prob_act

    #  -------- Save/Load network parameters --------
    def return_net_params(self):
        return lasagne.layers.helper.get_all_param_values(self.l_out)

    def set_net_params(self, net_params):
        lasagne.layers.helper.set_all_param_values(self.l_out, net_params)


# ===================================
# build neural network
# ===================================


def build_pg_network(input_height, input_width, output_length):
# input_height = time_horizon, input_width = (self.res_slot + self.max_job_size * self.num_nw) * self.num_res + self.backlog_width + 1
# output_length = self.num_nw + 1
#the network has a softmax output layer
    l_in = lasagne.layers.InputLayer(
        shape=(None, 1, input_height, input_width),
    )

    l_hid = lasagne.layers.DenseLayer(
        l_in,
        num_units=20,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hid,
        num_units=output_length,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.Normal(.01),
        b=lasagne.init.Constant(0)
    )

    return l_out
