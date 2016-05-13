from input_peatones_data  import DataSetManager
from numpy import *
import tensorflow as tf
from config import *
from squeezenet import SqueezeNet
import time

import os


config = ConfigMain()

dataset = DataSetManager(config.training_path,config.positive_windows,config.negative_windows,config.input_size,config.network_size)

#mages, labels =dataset.train.next_batch(4)



""" Input data initialization : As first lets just force resizing of the input """

test_data = tf.placeholder(tf.float32,shape=(config.batch_size, config.network_size[0],config.network_size[1],config.network_size[2]))
label_data = tf.placeholder(tf.int32,shape=(config.batch_size))
global_step = tf.Variable(0, trainable=False, name="global_step")

squeeze = SqueezeNet({'data':test_data})


""" Compute here the cross entropy function with logits ( LOSS FUNCTION) """

loss_function = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(squeeze.get_output(),label_data))

train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_function)





""" All the images that are going to tensorboard summary """

tf.image_summary('Input', test_data)
tf.scalar_summary('Loss', loss_function)
#tf.scalar_summary('Label', label_data)


summary_op = tf.merge_all_summaries()
sess = tf.InteractiveSession()


""" Set the saver """

saver = tf.train.Saver(tf.all_variables())


sess.run(tf.initialize_all_variables())


summary_writer = tf.train.SummaryWriter(config.summary_path,
                                            graph=sess.graph)

"""Load a previous model if restore is set to True"""

if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
ckpt = tf.train.get_checkpoint_state(config.models_path)
if config.restore:
  if ckpt:
    print 'Restoring from ', ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
  ckpt = 0


lowest_error = 10000;
lowest_iter = 0;


squeeze.load("squeeze.npy",sess)

if ckpt:
  initialIteration = int(ckpt.model_checkpoint_path.split('-')[1])
else:
  initialIteration = 1


for i in range(initialIteration, config.n_epochs*dataset.getNImagesDataset()):


	epoch_number = 1.0+ (float(i)*float(config.batch_size))/float(dataset.getNImagesDataset())
  
	""" Generate Images """
	batch = dataset.train.next_batch(config.batch_size)

	"""Save the model every 300 iterations"""
	if i%300 == 0:
		saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
		print 'Model saved.'

	start_time = time.time()
	test_output = sess.run(squeeze.get_output(), feed_dict={test_data: batch[0], label_data: batch[1]})
	print test_output
	print batch[1]

	#feedDict.update({test_data: batch[0], label_data: batch[1]})
	sess.run(train_step, feed_dict={test_data: batch[0], label_data: batch[1]})

	duration = time.time() - start_time


	if i%1 == 0:
		num_examples_per_step = config.batch_size
		examples_per_sec = num_examples_per_step / duration
		train_accuracy = sess.run(loss_function, feed_dict={test_data: batch[0], label_data: batch[1]})
		print train_accuracy
		if  train_accuracy < lowest_error:
			lowest_error = train_accuracy
			lowest_iter = i
		print("Epoch %f step %d, images used %d, loss %g, lowest_error %g on %d,examples per second %f"%(epoch_number, i, i*config.batch_size, train_accuracy, lowest_error, lowest_iter,examples_per_sec))

	""" Writing summary, not at every iterations """
	if i%2 == 0:

		#batch_val = dataset.validation.next_batch(config.batch_size)
		summary_str = sess.run(summary_op, feed_dict={test_data: batch[0], label_data: batch[1]})
		#summary_str_val,result= sess.run([val,last_layer], feed_dict=feedDict)
		summary_writer.add_summary(summary_str,i)

		