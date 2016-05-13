



def create_subnet_structure(tf, x,,net_data,permutation,sample_percentages):


	# The other convolutions however need to be replicated here.
	""" Convolution number one  """
	k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4  


	W_conv1 = tf.Variable(net_data["conv1"][0],name='Alex_W_conv1')

	W_b_conv1 = conv1b = tf.Variable(net_data["conv1"][1],name='Alex_W_b_conv1')

	
	conv1 = tf.nn.relu(conv(x, W_conv1, W_b_conv1, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1),name='Alex_W_conv1')



	""" Normalization number 1 """
	#lrn1
	#lrn(2, 2e-05, 0.75, name='norm1')
	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

	#maxpool1
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


	""" Convolution number two  """
	#conv(5, 5, 256, 1, 1, group=2, name='conv2')
	k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
	W_conv2 = tf.Variable(net_data["conv2"][0],name='Alex_W_conv2')
	W_b_conv2 = tf.Variable(net_data["conv2"][1],name='Alex_W_b_conv2')
	conv2 = tf.nn.relu(conv(maxpool1, W_conv2, W_b_conv2, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group),name='Alex_W_conv2')





	#lrn2
	#lrn(2, 2e-05, 0.75, name='norm2')
	radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
	lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius,alpha=alpha,beta=beta,bias=bias)

	#maxpool2
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)




	""" Convolution number three, The usual start point for creating substructures """
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1

	""" Take a weigth from alexnet and subsample it """


	weight_alex = tf.Variable(net_data["conv3"][0])
	bias_alex = tf.Variable(net_data["conv3"][1])

	# The new number of feature maps after sampling
	new_number_feat_maps = int(sample_percentages[0]*c_o)


	weight_positions = random.sample(range(0, c_o-1), new_number_feat_maps) # the number of possibilities for this layer.
	# Declare an initial sampling of the alex net
	W_conv3 = weight_alex[:,:,:,weight_positions[0]]
	W_b_conv3 = bias_alex[:,:,:,weight_positions[0]]
	del weight_positions[0]

	W_conv3 = tf.expand_dims(W_conv3, 3, name=None)
	W_b_conv3 = tf.expand_dims(W_b_conv3, 3, name=None)

	for i in weight_positions:
		""" The kernel shape is maintained but less kernels are taken"""
		W_conv3 = tf.concat(3,[W_conv3,tf.expand_dims(weight_alex[:,:,:,i],3,name=None)],name='Alex_W_conv3')
		W_b_conv3 = tf.concat(3,[W_b_conv3,tf.expand_dims(bias_alex[:,:,:,i],3,name=None)],name='Alex_W_b_conv3')


	conv3 = tf.nn.relu(conv(maxpool2, W_conv3, W_b_conv3, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group),name='Alex_conv3')

	

	""" Convolution number four  """
	#conv(3, 3, 384, 1, 1, group=2, name='conv4')
	k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2

	weight_alex = tf.Variable(net_data["conv4"][0])
	bias_alex = tf.Variable(net_data["conv4"][1])

	# The new number of feature maps after sampling
	new_number_feat_maps = int(sample_percentages[1]*c_o)


	weight_positions = random.sample(range(0, c_o-1), new_number_feat_maps) # the number of possibilities for this layer.
	# Declare an initial sampling of the alex net
	W_conv4 = weight_alex[:,:,:,weight_positions[0]]
	W_b_conv4 = bias_alex[:,:,:,weight_positions[0]]
	del weight_positions[0]

	W_conv4 = tf.expand_dims(W_conv4, 3, name=None)
	W_b_conv4 = tf.expand_dims(W_b_conv4, 3, name=None)

	for i in weight_positions:
		""" The kernel shape is maintained but less kernels are taken"""
		W_conv4 = tf.concat(3,[W_conv4,tf.expand_dims(weight_alex[:,:,:,i],3,name=None)],name='Alex_W_conv4')
		W_b_conv4 = tf.concat(3,[W_b_conv4,tf.expand_dims(bias_alex[:,:,:,i],3,name=None)],name='Alex_W_b_conv4')

	conv4 = tf.nn.relu(conv(conv3, W_conv4, W_b_conv4, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group),name='Alex_conv4')



	""" Convolution number five  """
	#conv5
	#conv(3, 3, 256, 1, 1, group=2, name='conv5')
	k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
	weight_alex = tf.Variable(net_data["conv5"][0])
	bias_alex = tf.Variable(net_data["conv5"][1])

	# The new number of feature maps after sampling
	new_number_feat_maps = int(sample_percentages[2]*c_o)

	# MAKE A FUNCTION FOR THIS PLEASE
	weight_positions = random.sample(range(0, c_o-1), new_number_feat_maps) # the number of possibilities for this layer.
	# Declare an initial sampling of the alex net
	W_conv5 = weight_alex[:,:,:,weight_positions[0]]
	W_b_conv5 = bias_alex[:,:,:,weight_positions[0]]
	del weight_positions[0]

	W_conv5 = tf.expand_dims(W_conv5, 3, name=None)
	W_b_conv5 = tf.expand_dims(W_b_conv5, 3, name=None)

	for i in weight_positions:
		""" The kernel shape is maintained but less kernels are taken"""
		W_conv5 = tf.concat(3,[W_conv5,tf.expand_dims(weight_alex[:,:,:,i],3,name=None)],name='Alex_W_conv5')
		W_b_conv5 = tf.concat(3,[W_b_conv5,tf.expand_dims(bias_alex[:,:,:,i],3,name=None)],name='Alex_W_b_conv5')

	conv5 = tf.nn.relu(conv(conv4, W_conv5, W_b_conv5, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group),name='Alex_conv5')



	""" Second Max-Pool  """

	#maxpool5
	#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
	k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
	maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)



    """ First fully connected ! The sixt layer of alexnet """
	#fc6
	#fc(4096, name='fc6')
	number_neurons = 4096

	weight_alex = tf.Variable(net_data["fc6"][0])
	bias_alex = tf.Variable(net_data["fc6"][1])

	# The new number of feature maps after sampling
	new_number_feat_maps = int(sample_percentages[3]*number_neurons)


	weight_positions = random.sample(range(0, c_o-1), new_number_feat_maps) # the number of possibilities for this layer.
	# Declare an initial sampling of the alex net
	W_fc6 = weight_alex[:,:,:,weight_positions[0]]
	W_b_fc6 = bias_alex[:,:,:,weight_positions[0]]
	del weight_positions[0]

	W_fc6 = tf.expand_dims(W_fc6, 3, name=None)
	W_b_fc6 = tf.expand_dims(W_b_fc6, 3, name=None)

	for i in weight_positions:
		""" The kernel shape is maintained but less kernels are taken"""
		W_fc6 = tf.concat(3,[W_fc6,tf.expand_dims(weight_alex[:,:,:,i],3,name=None)],name='Alex_W_fc6')
		W_b_fc6 = tf.concat(3,[W_b_fc6,tf.expand_dims(bias_alex[:,:,:,i],3,name=None)],name='Alex_W_b_fc6')



	fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), W_fc6, W_b_fc6)




	""" Second fully connected ! The seventh layer of alexnet """

	#fc7
	number_neurons = 4096

	weight_alex = tf.Variable(net_data["fc7"][0])
	bias_alex = tf.Variable(net_data["fc7"][1])

	# The new number of feature maps after sampling
	new_number_feat_maps = int(sample_percentages[3]*number_neurons)


	weight_positions = random.sample(range(0, c_o-1), new_number_feat_maps) # the number of possibilities for this layer.
	# Declare an initial sampling of the alex net
	W_fc7 = weight_alex[:,:,:,weight_positions[0]]
	W_b_fc7 = bias_alex[:,:,:,weight_positions[0]]
	del weight_positions[0]

	W_fc7 = tf.expand_dims(W_fc7, 3, name=None)
	W_b_fc7 = tf.expand_dims(W_b_fc7, 3, name=None)

	for i in weight_positions:
		""" The kernel shape is maintained but less kernels are taken"""
		W_fc7 = tf.concat(3,[W_fc7,tf.expand_dims(weight_alex[:,:,:,i],3,name=None)],name='Alex_W_fc7')
		W_b_fc7 = tf.concat(3,[W_b_fc7,tf.expand_dims(bias_alex[:,:,:,i],3,name=None)],name='Alex_W_b_fc7')




	#fc(4096, name='fc7')

	# THIS PART MAYBE IS NOT WORKING
	fc7 = tf.nn.relu_layer(fc6, W_fc7, W_b_fc7)

	""" Last Fully Connected, It is modified in order to be related to pedestrian """


	# CHECK AN EXAMPLE OF A FULLY CONNECTED shape defiinition 
	#fc(1000, relu=False, name='fc8')
    initializer = tf.uniform_unit_scaling_initializer(factor=1.15)
    W_fc8 = tf.get_variable(name='Alex_W_fc8', shape=[], initializer=initializer, trainable=True)
    initializer = tf.constant(0.1, shape=[])
    W_b_fc8 = tf.Variable(initial,name='Alex_W_b_fc8')

  

	fc8 = tf.nn.xw_plus_b(fc7, W_fc8, W_b_fc8,name='Alex_fc8')



