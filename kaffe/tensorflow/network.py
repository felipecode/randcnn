import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'


def layer(op):

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=True):
        data_dict = np.load(data_path).item()

        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    try:
                        print subkey
                        print key
                        print data.shape
                        var = tf.get_variable(subkey) # Here it assumes that the network is already mounted in some way, we need to break it.
                        # Get the shape
                        # Put data on correct spots ( Does it matter ??)
                        # How do I get the correct spots ??
                        # I Need to build the network while i am loading.

                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        # Do the variables already have names ????
        assert len(args) != 0
        self.inputs = []
        for fed_layer in args:
            print fed_layer
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.inputs.append(fed_layer)
        return self

    def get_output(self):
        return self.inputs[-1]

    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        return tf.get_variable(name, shape, trainable=self.trainable)

    def make_new_var_scaling(self, name, shape):
        initializer = tf.uniform_unit_scaling_initializer(factor=1.15)
        initial = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=True)
        return initial
    def make_new_bias_variable(self, shape):  
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             data,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1):

        #print padding

        data_dict = np.load(data_path).item()

        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        #print s_h,s_w
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        #print convolve
        #print 'passa'

        with tf.variable_scope(name) as scope:
            #kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            #biases = self.make_var('biases', [c_o])
            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                return tf.nn.relu(bias, name=scope.name)
            

            return tf.reshape(
                tf.nn.bias_add(conv, biases),
                conv.get_shape().as_list(),
                name=scope.name)


    @layer
    def conv_treinable(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1):

        #print padding
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        #print s_h,s_w
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        #print convolve
        #print 'passa'

        with tf.variable_scope(name) as scope:
            #kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            #biases = self.make_var('biases', [c_o])



            kernel =self.make_new_var_scaling('weights', shape=[k_h, k_w, c_i / group, c_o])
            biases = self.make_new_bias_variable( [c_o])
            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                return tf.nn.relu(bias, name=scope.name)
            

            return tf.reshape(
                tf.nn.bias_add(conv, biases),
                conv.get_shape().as_list(),
                name=scope.name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [int(input_shape[0]), dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        #print input 
        #print input.eval.shape
        #print 
        input = tf.reshape(input,[int(input.get_shape()[0]),int(input.get_shape()[3])])
        return tf.nn.softmax(input, name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
