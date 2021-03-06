"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import os
import sys
import timeit
import numpy
import time
import theano
import theano.tensor as T
from logistic_sgd import load_data
from mlp import MLP


def params_shape_like_shared(params):
    l = []
    for k in params:
        l.append(theano.shared(numpy.zeros_like(k.get_value(True))))
    return l


def params_shape_like(params):
    l = []
    for k in params:
        l.append(numpy.zeros_like(k.get_value(True)))
    return l

def params_shape_like_noise(params):
    l = []
    numpy.random.seed(1)
    for k in params:
        l.append(numpy.random.random(k.get_value(True).shape))
    return l

def set_parameters(clf, params):
    clf.hiddenLayer.W = theano.shared(params[0].get_value(True))
    clf.hiddenLayer.b = theano.shared(params[1].get_value(True))
    clf.logRegressionLayer.W = theano.shared(params[2].get_value(True))
    clf.logRegressionLayer.b = theano.shared(params[3].get_value(True))
    return clf


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2

def test_mlp_admm(learning_rate=0.5, L1_reg=0.00, L2_reg=0.0001, n_epochs=30000,
                  dataset='mnist.pkl.gz', batch_size=1000, n_hidden=500, rho=0.05):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """

    f = open('result.txt', 'w')
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels
    hw1 = T.matrix('hw1')
    hb1 = T.dvector('hb1')
    lw = T.matrix('lw')
    lb = T.dvector('hb')

    a1_hw1 = T.matrix('a1_hw1')
    a1_hb1 = T.dvector('a1_hb1')
    a1_lw = T.matrix('a1_lw')
    a1_lb = T.dvector('a1_hb')

    a2_hw1 = T.matrix('a2_hw1')
    a2_hb1 = T.dvector('a2_hb1')
    a2_lw = T.matrix('a2_lw')
    a2_lb = T.dvector('a2_hb')

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    update_model = theano.function(
        inputs=[hw1, hb1, lw, lb],
        updates=[(param, uparam)
                 for param, uparam in zip(classifier.params, [hw1, hb1, lw, lb])
                 ]
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    classifier_s = []
    cost_s = []
    train_model_s = []
    gparams_s = []
    updates_s = []
    w_params_s = []
    for i in range(n_train_batches):
        classifier_s.append(MLP(
            rng=rng,
            input=x,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10)
        )
        w_params_s.append(params_shape_like(classifier_s[i].params))
        cost_s.append(classifier_s[i].negative_log_likelihood(y)
                      + L1_reg * classifier_s[i].L1
                      + L2_reg * classifier_s[i].L2_sqr
                      + 0.5 * rho * classifier_s[i].augment([a1_hw1, a1_hb1, a1_lw, a1_lb],
                                                            [a2_hw1, a2_hb1, a2_lw, a2_lb])
                      )
        gparams_s.append([T.grad(cost_s[i], param) for param in classifier_s[i].params])
        updates_s.append([(param, param - learning_rate * gparam)
                          for param, gparam in zip(classifier_s[i].params, gparams_s[i])
                          ])
        train_model_s.append(theano.function(
            inputs=[index, a1_hw1, a1_hb1, a1_lw, a1_lb, a2_hw1, a2_hb1, a2_lw, a2_lb],
            outputs=cost_s[i],
            updates=updates_s[i],
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                # a1_hw1:classifier.params[0],
                # a1_hb1:classifier.params[1],
                # a1_lw:classifier.params[2],
                # a1_lb:classifier.params[3],
                # a2_hw1:w_params_s[i][0],
                # a2_hb1:w_params_s[i][1],
                # a2_lw:w_params_s[i][2],
                # a2_lb:w_params_s[i][3]
            })
        )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically

    # cost = (
    #     classifier.negative_log_likelihood(y)
    #     + L1_reg * classifier.L1
    #     + L2_reg * classifier.L2_sqr
    # )

    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams



    # gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]



    # updates = [
    #     (param, param - learning_rate * gparam)
    #     for param, gparam in zip(classifier.params, gparams)
    # ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`


    # train_model = theano.function(
    #     inputs=[index],
    #     outputs=cost,
    #     updates=updates,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: train_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )



    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = 1
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    update_params = params_shape_like(classifier.params)
    while (epoch < n_epochs) and (not done_looping):
        updating_start_time = time.time()
        epoch = epoch + 1
        tmp = params_shape_like(classifier.params)
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model_s[minibatch_index](minibatch_index,
                                                                update_params[0],
                                                                update_params[1],
                                                                update_params[2],
                                                                update_params[3],
                                                                w_params_s[minibatch_index][0],
                                                                w_params_s[minibatch_index][1],
                                                                w_params_s[minibatch_index][2],
                                                                w_params_s[minibatch_index][3])
            # print minibatch_avg_cost,
            iter = (epoch - 1) * n_train_batches + minibatch_index
            for p_index in range(len(classifier.params)):
                tmp[p_index] += classifier_s[minibatch_index].params[p_index].get_value(True) + \
                               w_params_s[minibatch_index][p_index]


        for p_index in range(len(classifier.params)):
            update_params[p_index] = tmp[p_index] / n_train_batches

        for minibatch_index in xrange(n_train_batches):
            for p_index in range(len(classifier_s[minibatch_index].params)):
                w_params_s[minibatch_index][p_index] += classifier_s[minibatch_index].params[p_index].get_value(True) - \
                                                        update_params[p_index]


        # update_model(update_params[0], update_params[1], update_params[2], update_params[3])

        update_model(update_params[0], update_params[1], update_params[2], update_params[3])
        # print classifier.params[3].get_value(True)
        # print
        updating_time = time.time() - updating_start_time
        print 'updating time usage:', updating_time / n_train_batches
        f.writelines('updating time usage:' + str(updating_time / n_train_batches) + '\n')

        if epoch % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)

            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )
            )
            f.writelines('validation error:' + str(this_validation_loss * 100.) + '\n')

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                ):
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [test_model(i) for i
                               in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)

                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))
                f.writelines('test error:' + str(test_score * 100.) + '\n')

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp_admm()
