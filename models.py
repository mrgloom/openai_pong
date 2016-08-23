import lasagne
from lasagne.layers import dnn, batch_norm

def build_nature_with_pad3(n_actions, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 4, 80, 80),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(8, 8), stride=4, pad=4,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(.1))

    print(network.output_shape)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(4, 4), stride=2, pad=2,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    print(network.output_shape)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3), stride=1, pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    print(network.output_shape)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=128,
        nonlinearity=lasagne.nonlinearities.rectify,
        b=lasagne.init.Constant(.1))

    network = lasagne.layers.DenseLayer(
        network,
        num_units=n_actions,
        nonlinearity=None,
        b=lasagne.init.Constant(.1))

    return network

def build_nature_dnn(n_actions, input_var):
    from lasagne.layers import dnn

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 4, 80, 80),
        input_var=input_var
    )

    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=32,
        filter_size=(8, 8),
        stride=(4, 4),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv1 = batch_norm(l_conv1)

    l_conv2 = dnn.Conv2DDNNLayer(
        l_conv1,
        num_filters=64,
        filter_size=(4, 4),
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_conv3 = dnn.Conv2DDNNLayer(
        l_conv2,
        num_filters=64,
        filter_size=(3, 3),
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv3,
        num_units=256,
        #num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=n_actions,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1)
    )

    return l_out


